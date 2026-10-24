"""InternVL3 attention extraction.

Frame branch: InternVL3 splits a frame into 448 x 448 tiles chosen to match
its aspect ratio ("dynamic tiling") plus a global thumbnail, and turns every
tile into a 16 x 16 token grid (pixel shuffle). With at most 4 tiles the frame
stays within the 1024-token budget. The tile grids are stitched back into one
``(16 * rows, 16 * cols)`` map; the thumbnail tokens are ignored.

Video branch: every frame is encoded as a single 448 x 448 tile, giving a fixed
16 x 16 grid (256 tokens) per frame; InternVL3 has no smaller video setting.
"""

from __future__ import annotations

from typing import List

import torch
from PIL import Image
from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from atlas.lvlm.base import AttentionExtractor

MAX_TILES = 4


class InternVLExtractor(AttentionExtractor):

    @staticmethod
    def apply_rope(module, q, k, cos, sin):
        return apply_rotary_pos_emb(q, k, cos, sin)

    def _inputs(self, kind: str, prompt: str, **media):
        messages = [{"role": "user", "content": [{"type": kind}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        return self.processor(text=[text], return_tensors="pt", **media)

    @property
    def tile_side(self) -> int:
        return int(round(self.processor.image_seq_length ** 0.5))  # 16

    def frame_attention(self, image_path: str, prompt: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        inputs = self._inputs("image", prompt, images=[image], max_patches=MAX_TILES)
        attn = self.run(inputs, self.processor.image_token_id)
        size = self.processor.image_processor.size
        cols, rows = get_optimal_tiled_canvas(
            (image.height, image.width), (size["height"], size["width"]), 1, MAX_TILES,
        )
        s = self.tile_side
        tiles = rows * cols
        assert attn.shape[1] == s * s * (tiles + (tiles > 1)), "unexpected InternVL tile layout"
        grid = attn[:, : tiles * s * s].reshape(self.channels, rows, cols, s, s)
        return grid.permute(0, 1, 3, 2, 4).reshape(self.channels, rows * s, cols * s)

    def video_attention(self, frame_paths: List[str], prompt: str) -> torch.Tensor:
        frames = [Image.open(p).convert("RGB") for p in frame_paths]
        attn = self.run(self._inputs("video", prompt, videos=[frames]), self.processor.image_token_id)
        s = self.tile_side
        assert attn.shape[1] == len(frames) * s * s, "unexpected InternVL video token layout"
        return attn.reshape(self.channels, len(frames), s, s).transpose(0, 1)
