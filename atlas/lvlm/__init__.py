"""Attention extractors for the frozen LVLMs used in the paper."""

from atlas.config import lvlm_path


def load_extractor(name: str, device: str = "cuda", attn_implementation: str = "flash_attention_2"):
    """Build the attention extractor for an LVLM short name (see ``atlas.config.LVLMS``)."""
    if name.startswith("qwen"):
        from atlas.lvlm.qwen import QwenVLExtractor as cls
    elif name.startswith("llava-ov"):
        from atlas.lvlm.llava_ov import LlavaOneVisionExtractor as cls
    elif name.startswith("internvl"):
        from atlas.lvlm.internvl import InternVLExtractor as cls
    else:
        raise ValueError(f"unknown LVLM {name!r}")
    return cls(lvlm_path(name), device=device, attn_implementation=attn_implementation)
