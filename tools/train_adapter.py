"""Stage 2 - train the attention adaptation network (Sec. 3.3, Sec. 4, Appendix B).

Trains only the adapter on Ref-YouTube-VOS train attention caches written by
``tools/extract_attention.py``; the LVLM and SAM2 stay frozen. Each training
example is one clip (all keyframes of a video-expression pair); a batch holds
8 clips. Loss: BCE-with-logits + soft Dice against the ground-truth masks
(Eq. 10). AdamW, learning rate decayed linearly from 1e-4 to 1e-5 over 20
epochs, weight decay 1e-4.
"""

import argparse
import csv
import glob
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from atlas.adapter import AttentionAdapter, adapter_input


class AttentionCaches(Dataset):
    """Adapter inputs ``(T, 2C, H, W)`` and soft targets ``(T, H, W)`` on a fixed grid."""

    def __init__(self, paths, grid):
        self.paths, self.grid = paths, grid

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rec = torch.load(self.paths[i], map_location="cpu")
        x = adapter_input(rec["frame_attn"], rec["video_attn"], self.grid)
        y = F.interpolate(rec["gt_frame"].float()[:, None], size=self.grid, mode="bilinear", align_corners=False)
        return x, y[:, 0].clamp(0, 1)


def collate(batch):
    """Pad clips to the longest one; targets of the valid keyframes are concatenated."""
    lengths = [x.shape[0] for x, _ in batch]
    xs = torch.zeros(len(batch), max(lengths), *batch[0][0].shape[1:])
    for i, (x, _) in enumerate(batch):
        xs[i, : x.shape[0]] = x
    return xs, torch.cat([y for _, y in batch]), lengths


def dice_loss(prob, target, eps=1.0):
    inter = (prob * target).sum()
    return 1 - (2 * inter + eps) / (prob.sum() + target.sum() + eps)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True, help="Ref-YouTube-VOS train attention caches.")
    p.add_argument("--out", required=True, help="Output folder for checkpoints and logs.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8, help="Clips per batch.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="", help="Checkpoint saved by an earlier run of this script.")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.cache, "*.pt")))
    probe = torch.load(paths[0], map_location="cpu")
    grid = tuple(probe["frame_attn"].shape[-2:])  # all clips are resized to this grid
    in_channels = 2 * probe["frame_attn"].shape[1]
    print(f"{len(paths)} clips | grid {grid} | input channels {in_channels}")

    loader = DataLoader(
        AttentionCaches(paths, grid), batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = AttentionAdapter(in_channels).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    log = open(os.path.join(args.out, "train_log.csv"), "a", newline="")
    writer = csv.writer(log)
    for epoch in range(start_epoch, args.epochs):
        # Linear decay from lr (first epoch) to lr_min (last epoch).
        lr = args.lr + (args.lr_min - args.lr) * epoch / max(args.epochs - 1, 1)
        for group in opt.param_groups:
            group["lr"] = lr
        model.train()
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs} lr={lr:.1e}")
        for step, (x, y, lengths) in enumerate(bar):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = model(x, lengths)
            bce = F.binary_cross_entropy_with_logits(logits, y)
            dice = dice_loss(logits.sigmoid(), y)
            loss = bce + dice
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bar.set_postfix(loss=f"{loss.item():.3f}")
            writer.writerow([epoch + 1, step, f"{loss.item():.5f}", f"{bce.item():.5f}", f"{dice.item():.5f}"])
        log.flush()

        torch.save({"model": model.state_dict(), "config": model.config, "grid": grid,
                    "optimizer": opt.state_dict(), "epoch": epoch},
                   os.path.join(args.out, "adapter_last.pt"))
    torch.save({"model": model.state_dict(), "config": model.config, "grid": grid},
               os.path.join(args.out, "adapter.pt"))
    print(f"saved {os.path.join(args.out, 'adapter.pt')}")


if __name__ == "__main__":
    main()
