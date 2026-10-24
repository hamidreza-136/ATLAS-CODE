"""Qwen2-VL / Qwen2.5-VL attention extraction.

Both models resize each image with ``smart_resize`` under a ``max_pixels``
budget (28 x 28 pixels per visual token, aspect ratio preserved) and merge
2 x 2 patches into one token, so the visual tokens of an image form an
``(h, w)`` grid in row-major order.

Video branch: Qwen groups ``temporal_patch_size = 2`` consecutive frames into
one temporal token. Each keyframe is therefore fed twice, so the video
attention has exactly one temporal slot per keyframe (``T_v = T_s``).
"""

from __future__ import annotations

from typing import List

import torch
from PIL import Image
from qwen_vl_utils import fetch_image, process_vision_info
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

from atlas.config import FRAME_TOKENS, VIDEO_TOKENS
from atlas.lvlm.base import AttentionExtractor

PIXELS_PER_TOKEN = 28 * 28


def _pad_to_common_canvas(images: List[Image.Image]) -> List[Image.Image]:
    """Center-pad frames to the largest size so they can be stacked as a video."""
    w = max(im.size[0] for im in images)
    h = max(im.size[1] for im in images)
    if all(im.size == (w, h) for im in images):
        return images
    out = []
    for im in images:
        canvas = Image.new("RGB", (w, h))
        canvas.paste(im, ((w - im.size[0]) // 2, (h - im.size[1]) // 2))
        out.append(canvas)
    return out


class QwenVLExtractor(AttentionExtractor):

    @staticmethod
    def apply_rope(module, q, k, cos, sin):
        return apply_multimodal_rotary_pos_emb(q, k, cos, sin, module.rope_scaling["mrope_section"])

    def processor_kwargs(self) -> dict:
        return {"use_fast": False}

    def _inputs(self, content, prompt):
        messages = [{"role": "user", "content": [content, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        return self.processor(text=[text], images=images, videos=videos, return_tensors="pt")

    def frame_attention(self, image_path: str, prompt: str) -> torch.Tensor:
        content = {"type": "image", "image": image_path, "max_pixels": FRAME_TOKENS * PIXELS_PER_TOKEN}
        inputs = self._inputs(content, prompt)
        attn = self.run(inputs, self.processor.image_token_id)
        merge = self.processor.image_processor.merge_size
        _, gh, gw = inputs["image_grid_thw"][0].tolist()
        return attn.reshape(self.channels, gh // merge, gw // merge)

    def video_attention(self, frame_paths: List[str], prompt: str) -> torch.Tensor:
        frames = [fetch_image({"image": p, "max_pixels": VIDEO_TOKENS * PIXELS_PER_TOKEN}) for p in frame_paths]
        frames = _pad_to_common_canvas(frames)
        repeat = self.processor.image_processor.temporal_patch_size
        frames = [f for f in frames for _ in range(repeat)]
        inputs = self._inputs({"type": "video", "video": frames}, prompt)
        attn = self.run(inputs, self.processor.video_token_id)
        merge = self.processor.image_processor.merge_size
        t, gh, gw = inputs["video_grid_thw"][0].tolist()
        return attn.reshape(self.channels, t, gh // merge, gw // merge).transpose(0, 1)
