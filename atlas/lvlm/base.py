"""Shared machinery for extracting LVLM attention maps (Sec. 3.2).

For a prompt that contains visual tokens, ATLAS needs the attention of the last
input token (the position that generates the single-word answer) to every
visual token, for every head of every layer from the middle layer ``L0 = L/2``
to the last layer. Stacking them gives a ``(C, h, w)`` tensor with
``C = (L - L0) * H`` channels (Eq. 1 and 2).

The model itself runs with a fast attention kernel. A forward pre-hook on each
selected attention module recomputes the rotary-embedded queries and keys and
evaluates only the last query row in fp32, so no full ``L x L`` attention
matrix is ever materialized.
"""

from __future__ import annotations

import inspect
from typing import Callable, List

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen2.modeling_qwen2 import repeat_kv


class LastTokenAttention:
    """Records the last-token attention ``(H, L)`` of each hooked decoder layer."""

    def __init__(self, attn_modules: List[torch.nn.Module], apply_rope: Callable):
        self.apply_rope = apply_rope
        self.maps: List[torch.Tensor] = []
        self.handles = [
            m.register_forward_pre_hook(self._hook, with_kwargs=True) for m in attn_modules
        ]

    @torch.no_grad()
    def _hook(self, module, args, kwargs):
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        cos, sin = kwargs["position_embeddings"]
        bsz, seq_len, _ = hidden.shape
        hd = module.head_dim
        q = module.q_proj(hidden).view(bsz, seq_len, -1, hd).transpose(1, 2)
        k = module.k_proj(hidden).view(bsz, seq_len, -1, hd).transpose(1, 2)
        q, k = self.apply_rope(module, q, k, cos, sin)
        k = repeat_kv(k, module.num_key_value_groups)
        # Causal attention of the last position covers every token of the prompt.
        logits = q[:, :, -1:].float() @ k.float().transpose(-1, -2) / hd ** 0.5
        self.maps.append(logits.softmax(dim=-1)[0, :, 0])  # (H, L)

    def pop(self) -> torch.Tensor:
        """Return the recorded maps stacked as ``(num_layers * H, L)`` and reset."""
        out = torch.cat(self.maps, dim=0)
        self.maps = []
        return out


class AttentionExtractor:
    """Frozen LVLM wrapper returning frame-level and video-level attention maps.

    Subclasses implement ``frame_attention`` and ``video_attention`` by building
    the model-specific inputs and mapping visual-token positions to a spatial
    grid. ``apply_rope`` must reproduce the rotary embedding of the model.
    """

    apply_rope: Callable = None

    def __init__(self, model_path: str, device: str = "cuda", attn_implementation: str = "flash_attention_2"):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map=device,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, **self.processor_kwargs())
        self.device = self.model.device

        layers = self.model.model.language_model.layers
        self.num_layers = len(layers)
        self.first_layer = self.num_layers // 2  # L0: middle transformer layer
        self.num_heads = getattr(self.model.config, "text_config", self.model.config).num_attention_heads
        self.recorder = LastTokenAttention(
            [layer.self_attn for layer in layers[self.first_layer:]], type(self).apply_rope,
        )
        params = inspect.signature(self.model.forward).parameters
        self.forward_kwargs = {"logits_to_keep": 1} if "logits_to_keep" in params else {}

    @property
    def channels(self) -> int:
        return (self.num_layers - self.first_layer) * self.num_heads

    def processor_kwargs(self) -> dict:
        return {}

    @torch.no_grad()
    def run(self, inputs, visual_token_id: int):
        """Forward pass; returns the ``(C, V)`` attention of the last token to the visual tokens."""
        inputs = inputs.to(self.device, dtype=self.model.dtype)  # casts pixel values only
        self.model(**inputs, use_cache=False, **self.forward_kwargs)
        attn = self.recorder.pop()
        visual = inputs["input_ids"][0] == visual_token_id
        return attn[:, visual].float()

    def frame_attention(self, image_path: str, prompt: str):
        """``(C, h, w)`` attention for one keyframe at the frame-branch token budget."""
        raise NotImplementedError

    def video_attention(self, frame_paths: List[str], prompt: str):
        """``(T, C, h, w)`` attention for all keyframes in one pass (video branch)."""
        raise NotImplementedError
