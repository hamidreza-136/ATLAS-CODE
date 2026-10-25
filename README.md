# ATLAS: Learning to Adapt LVLM Attention for Reasoning Video Segmentation

Hamidreza Dastmalchi, Ali Cheraghian, Aijun An, Abdelwahed Khamis, Lars Petersson
**WACV 2027** · [arXiv:2610.24783](https://arxiv.org/abs/2610.24783) · [Project page](https://hamidreza-136.github.io/atlas/)

ATLAS turns the raw attention of a frozen large vision-language model (LVLM)
into object-centric localization maps with a lightweight adapter, then uses
them to prompt a frozen SAM2 that segments and propagates the referred object
through the video. Only the adapter (~15M parameters) is trained.

| Stage (paper section) | What happens | Code |
|---|---|---|
| 1. Attention extraction (3.2) | Frame-level (1024 tokens, one pass per keyframe) and video-level (64 tokens, one pass over all keyframes) attention of the answer token to the visual tokens, all heads of layers L/2..L | [`atlas/lvlm/`](atlas/lvlm), [`tools/extract_attention.py`](tools/extract_attention.py) |
| 2. Attention adaptation (3.3) | U-Net + temporal Transformer + FiLM, trained with BCE + Dice on Ref-YouTube-VOS | [`atlas/adapter.py`](atlas/adapter.py), [`tools/train_adapter.py`](tools/train_adapter.py) |
| 3. Segmentation and propagation (3.4) | Argmax point prompts, duplicate-seed removal by propagation, per-frame Pearson tracklet selection, SAM2 propagation | [`atlas/segmentation.py`](atlas/segmentation.py), [`tools/segment_and_propagate.py`](tools/segment_and_propagate.py) |

Supported LVLMs: `qwen2.5-vl-7b`, `qwen2-vl-7b`, `llava-ov-7b`, `internvl3-8b`.
Benchmarks: Ref-DAVIS17, ReasonVOS, ReVOS, MeViS, Ref-YouTube-VOS.

## Installation

```bash
git clone https://github.com/hamidreza-136/ATLAS-CODE.git
cd ATLAS-CODE
python3.12 -m venv .venv && source .venv/bin/activate

# PyTorch for your CUDA version (tested: 2.11.0 + CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# FlashAttention-2 (used for attention extraction; see the note below)
pip install flash-attn --no-build-isolation

# SAM2, pinned to the version used in the paper. --no-build-isolation builds
# its CUDA extension (needs nvcc), which SAM2 uses to fill small mask holes.
pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@2b90b9f"
```

> **FlashAttention.** The LVLMs run with `flash_attention_2`, as for the paper.
> Without flash-attn, pass `--attn-implementation sdpa` to
> `tools/extract_attention.py`. This works, but the bf16 vision encoders give
> attention maps that differ by a few percent, so use one setting for both
> training and evaluation caches.

## Checkpoints

All checkpoints go into `ckpts/` (override with `ATLAS_CKPT_ROOT`).

```bash
# Frozen LVLMs (only the ones you need)
hf download Qwen/Qwen2.5-VL-7B-Instruct             --local-dir ckpts/Qwen2.5-VL-7B-Instruct
hf download Qwen/Qwen2-VL-7B-Instruct               --local-dir ckpts/Qwen2-VL-7B-Instruct
hf download llava-hf/llava-onevision-qwen2-7b-ov-hf --local-dir ckpts/llava-onevision-qwen2-7b-ov-hf
hf download OpenGVLab/InternVL3-8B-hf               --local-dir ckpts/InternVL3-8B-hf

# Frozen SAM2-Hiera-Large
wget -O ckpts/sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

## Data

Datasets go into `datasets/` (override with `ATLAS_DATA_ROOT`):

```text
datasets/
├── ref-youtube-vos/   meta_expressions/{train,valid,test}/meta_expressions.json, {train,valid}/JPEGImages, train/Annotations
├── davis17/           meta_expressions/valid/meta_expressions.json, valid/{JPEGImages,mask_dict.pkl}
├── ReasonVOS/         meta_expressions_v2.json, mask_dict_v2.pkl, JPEGImages
├── ReVOS/             meta_expressions_valid_.json, mask_dict.json, JPEGImages
└── mevis/             valid_u/{meta_expressions.json,mask_dict.json,JPEGImages}, valid/{meta_expressions.json,JPEGImages}
```

```bash
mkdir -p datasets downloads

# Ref-DAVIS17 and ReasonVOS (packaged with mask dictionaries by DecAF)
hf download js-hyun/decaf_data --repo-type dataset --local-dir downloads/decaf_data
tar -xf downloads/decaf_data/RVOSJoint/davis17_data.tar   -C datasets
tar -xf downloads/decaf_data/RVOSJoint/reasonvos_data.tar -C datasets

# ReVOS (packaged with mask dictionaries by DecAF)
hf download CIPLab-Video/decaf-rvos-revos --repo-type dataset --local-dir downloads/revos
tar -xf downloads/revos/ReVOS.tar -C datasets

# MeViS (official release; only valid_u and valid are needed)
hf download FudanCVL/MeViS --repo-type dataset --local-dir downloads/mevis
unzip -q downloads/mevis/MeViS_release.zip -x "__MACOSX/*" "MeViS_release/train/*" -d downloads/mevis
mv downloads/mevis/MeViS_release datasets/mevis
for s in valid valid_u; do tar -xf datasets/mevis/$s/JPEGImages.tar -C datasets/mevis/$s && rm datasets/mevis/$s/JPEGImages.tar; done
```

Ref-YouTube-VOS (training set and validation set) is distributed through the
[official challenge page](https://youtube-vos.org/dataset/rvos/); place it in
`datasets/ref-youtube-vos/` with the layout above.

## Running

Every script takes the LVLM name; `GPUS` lists the GPUs to shard over.
Interrupted runs can be restarted: finished samples are skipped.

```bash
LVLM=qwen2.5-vl-7b      # or qwen2-vl-7b | llava-ov-7b | internvl3-8b

# 1. Attention extraction: training set, then any evaluation set
GPUS="0 1 2 3" bash scripts/1_extract_attention.sh $LVLM ytvos_train
GPUS="0 1 2 3" bash scripts/1_extract_attention.sh $LVLM davis

# 2. Train the adapter (20 epochs, one GPU) -> runs/$LVLM/adapter.pt
GPU=0 bash scripts/2_train_adapter.sh $LVLM

# 3. Adapt attention, segment and propagate with SAM2, evaluate J&F
GPUS="0 1 2 3" bash scripts/3_segment_and_propagate.sh $LVLM davis
```

Or run everything for one LVLM (training plus all six evaluation sets):

```bash
GPUS="0 1 2 3" bash scripts/reproduce.sh $LVLM
SKIP_TRAIN=1 GPUS="0 1 2 3" bash scripts/reproduce.sh $LVLM davis reasonvos   # reuse already-trained adapters
```

Evaluation sets: `davis`, `reasonvos`, `revos` (referring / reasoning / overall),
`mevis` (MeViS valid_u), `ytvos_valid` and `mevis_valid`. The last two have no
public ground truth: step 3 writes `outputs/<lvlm>/<dataset>_submission.zip` for
the [Ref-YouTube-VOS](https://codalab.lisn.upsaclay.fr/competitions/3282) and
[MeViS](https://github.com/henghuiding/MeViS) evaluation servers.

Outputs:

```text
cache/<lvlm>/<dataset>/<video>__<exp_id>.pt     attention caches (fp16, ~13 MB per sample)
runs/<lvlm>/adapter.pt                         trained adapter (+ adapter_last.pt, train_log.csv)
outputs/<lvlm>/<dataset>/<video>/<exp_id>/*.png predicted masks
outputs/<lvlm>/<dataset>.json / .csv           J, F, J&F (summary / per expression)
```

## Implementation details

**Keyframes and attention.** 16 uniformly sampled keyframes. The expression is
wrapped in the object extraction prompt of Appendix A
([`atlas/prompts.py`](atlas/prompts.py)), and the attention of the last prompt
token (which produces the one-word answer) to every visual token is taken
from all heads of layers `L/2 .. L-1`. All four 7-8B LVLMs have 28 layers with
28 heads, so `C = 14 x 28 = 392` channels per branch. The attention is
recomputed in fp32 from the model's own queries and keys.

**Visual-token budgets.** Each LVLM tokenizes images differently:

| LVLM | Frame branch | Video branch |
|---|---|---|
| Qwen2.5-VL / Qwen2-VL | ≤ 1024 tokens, aspect ratio preserved (e.g. 17 x 32 for 854 x 480) | ≤ 64 tokens per frame (e.g. 5 x 11); each keyframe is fed twice so that every keyframe gets its own temporal token |
| LLaVA-OneVision | AnyRes grid without padding, shrunk to about one tile (≤ ~880 tokens, e.g. 19 x 37) | native 14 x 14 = 196 tokens per frame (no smaller setting) |
| InternVL3 | up to 4 dynamic 448 px tiles of 16 x 16 tokens (≤ 1024 tokens, e.g. 16 x 32); thumbnail ignored | native 16 x 16 = 256 tokens per frame (no smaller setting) |

**Adapter** (Table 10). The video attention is bilinearly resized to the frame
grid and concatenated (784 input channels). During training every clip is
resized to the frame grid of the first training cache, which is stored in the
checkpoint and used at test time. Encoder: 3x3 conv x2 (128) → 2x2 max-pool →
3x3 conv x2 (256) → bottleneck 3x3 conv x2 (512), each conv followed by
BatchNorm + ReLU. The globally pooled bottleneck features of the keyframes go
through a 2-layer, 4-head Transformer; a linear layer maps its outputs to FiLM
parameters `(γ, β)`, and `Z' = Z ⊙ (1 + γ) + β`. Decoder: transposed conv +
skip + conv x2 (256), transposed conv + skip + conv x2 (128), 1x1 conv to
logits. Training: AdamW, batch of 8 clips, learning rate 1e-4 → 1e-5 (linear
per epoch), weight decay 1e-4, 20 epochs, BCE-with-logits + soft Dice.

**Segmentation.** One positive point per keyframe at the argmax of `P_k`. Seeds
are visited from the earliest keyframe; a seed's mask is propagated over the
keyframes, and later seeds whose SAM2 mask overlaps the propagated mask with
IoU > 0.8 are dropped. Every kept seed is one candidate tracklet. The tracklet
with the highest mean per-keyframe Pearson correlation between its
(average-pooled) mask probabilities and `P_k` is propagated over the full
video. If that score is below 0.1, the prediction is empty.

## Citation

```bibtex
@article{dastmalchi2026atlas,
  title={ATLAS: Learning to Adapt LVLM Attention for Reasoning Video Segmentation},
  author={Dastmalchi, Hamidreza and Cheraghian, Ali and An, Aijun and Khamis, Abdelwahed and Petersson, Lars},
  journal={arXiv preprint arXiv:2610.24783},
  year={2026}
}
```

## Acknowledgements

The evaluation protocol and dataset packaging follow
[DecAF](https://github.com/HYUNJS/DecAF); masks are produced with
[SAM 2](https://github.com/facebookresearch/sam2).
