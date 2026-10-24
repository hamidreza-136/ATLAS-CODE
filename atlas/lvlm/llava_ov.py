"""LLaVA-OneVision attention extraction.

Frame branch: LLaVA-OV encodes a frame as a 27 x 27 overview of the whole
image followed by an "AnyRes" high-resolution grid (tiles stitched together,
padding removed, one newline token after every row). The high-resolution grid
is used as the spatial map. ``vision_aspect_ratio = "anyres_max_1"`` makes the
model bilinearly shrink that grid to about one tile (27 x 27 = 729 tokens, at
most 1.1^2 x 729), i.e. within the 1024-token frame budget, while keeping the
aspect ratio.

Video branch: every frame is encoded at 384 x 384 and pooled to a fixed
14 x 14 grid (196 tokens per frame); LLaVA-OV has no smaller video setting,
so this native budget replaces the 64-token budget used for Qwen. One newline
token follows the last frame.
"""

from __future__ import annotations

from typing import List

import torch
from PIL import Image
from transformers.image_processing_utils import select_best_resolution
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from atlas.lvlm.base import AttentionExtractor

ANYRES = "anyres_max_1"


class LlavaOneVisionExtractor(AttentionExtractor):

    @staticmethod
    def apply_rope(module, q, k, cos, sin):
        return apply_rotary_pos_emb(q, k, cos, sin)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Processor and model must agree on the AnyRes budget (token count).
        self.processor.vision_aspect_ratio = ANYRES
        self.model.config.vision_aspect_ratio = ANYRES
        vision = self.model.config.vision_config
        self.base_side = vision.image_size // vision.patch_size  # 27

    def _inputs(self, kind: str, prompt: str, **media):
        messages = [{"role": "user", "content": [{"type": kind}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        return self.processor(text=[text], return_tensors="pt", **media)

    def _anyres_grid(self, height: int, width: int):
        """(rows, cols) of the unpadded, resized AnyRes grid for an image of this size."""
        best_h, best_w = select_best_resolution([height, width], self.processor.image_processor.image_grid_pinpoints)
        tile = self.model.config.vision_config.image_size
        n_tokens, rows = self.processor._get_unpadded_features(
            height, width, self.base_side, self.base_side, best_h // tile, best_w // tile,
        )
        return rows, n_tokens // rows

    def frame_attention(self, image_path: str, prompt: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        attn = self.run(self._inputs("image", prompt, images=[image]), self.processor.image_token_id)
        rows, cols = self._anyres_grid(image.height, image.width)
        hires = attn[:, self.base_side ** 2:]  # drop the 27 x 27 overview
        assert hires.shape[1] == rows * (cols + 1), "unexpected LLaVA-OV image token layout"
        return hires.reshape(self.channels, rows, cols + 1)[:, :, :cols]  # drop newline tokens

    def video_attention(self, frame_paths: List[str], prompt: str) -> torch.Tensor:
        frames = [Image.open(p).convert("RGB") for p in frame_paths]
        attn = self.run(self._inputs("video", prompt, videos=[frames]), self.processor.video_token_id)
        t = len(frames)
        side = (attn.shape[1] - 1) // t
        side = int(round(side ** 0.5))
        attn = attn[:, : t * side * side]  # drop the trailing newline token
        return attn.reshape(self.channels, t, side, side).transpose(0, 1)
