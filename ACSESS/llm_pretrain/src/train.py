import os
import sys
import time
import json
import math
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, get_rank, get_world_size
import torch.distributed as dist

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from model import LlamaLM


def setup_ddp():
    if not dist.is_initialized():
        init_process_group(backend="nccl")
    torch.cuda.set_device(get_rank())


def cleanup_ddp():
    if dist.is_initialized():
        destroy_process_group()


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, token_file: str, seq_len: int, num_samples: int = None):
        self.data = torch.from_numpy(
            torch.load(token_file, map_location='cpu', weights_only=True).numpy()
            if token_file.endswith('.pt')
            else torch.tensor([int(x) for x in open(token_file).read().split()], dtype=torch.int32).numpy()
        ).to(dtype=torch.int32)
        if num_samples:
            self.data = self.data[:num_samples * seq_len]
        self.seq_len = seq_len
        self.num_samples = len(self.data) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        tokens = self.data[start : start + self.seq_len]
        input_ids = tokens[:-1].to(torch.long)
        labels = tokens[1:].to(torch.long)
        return input_ids, labels


class MemoryEfficientDataLoader:
    def __init__(self, data_dir: str, seq_len: int, batch_size: int, world_size: int, rank: int):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.shard_files = sorted([f for f in self.data_dir.glob("*.pt") if f.name.startswith("shard_")])
        self.current_shard_idx = 0
        self.current_loader = None
        self.samples_seen = 0

    def _load_current_shard(self):
        if self.current_shard_idx >= len(self.shard_files):
            return False
        shard_file = self.shard_files[self.current_shard_idx]
        dataset = TokenDataset(str(shard_file), self.seq_len)
        sampler = DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True
        )
        self.current_loader = DataLoader(
            dataset, batch_size=self.batch_size, sampler=sampler, num_workers=0
        )
        self.current_shard_idx += 1
        return True

    def __iter__(self):
        while self._load_current_shard():
            for batch in self.current_loader:
                yield batch
                self.samples_seen += 1


def get_lr(step: int, warmup_steps: int, total_steps: int, peak_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--peak_lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--total_steps", type=int, default=47700)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=60)
    parser.add_argument("--max_checkpoints", type=int, default=3)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_ddp()
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device("cuda")

    torch.manual_seed(args.seed + rank)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = LlamaLM(vocab_size=50304, d_model=1024, n_layers=16, n_heads=16, mlp_hidden=2816, max_seq_len=2048)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1
    )

    scaler = torch.amp.GradScaler("cuda")
    scheduler_step = 0
    start_step = 0
    start_time = time.time()
    tokens_seen = 0

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler_step = ckpt["scheduler_step"]
        start_step = ckpt["step"]
        tokens_seen = ckpt["tokens_seen"]
        torch.manual_seed(ckpt["seed"])
        if rank == 0:
            print(f"Resumed from step {start_step}, tokens seen: {tokens_seen}")

    if rank == 0 and SummaryWriter:
        writer = SummaryWriter(str(output_dir / "logs"))
    else:
        writer = None

    csv_path = output_dir / "metrics.csv"
    if not csv_path.exists() and rank == 0:
        csv_path.write_text("step,loss,lr,tokens_per_sec,mfu,grad_scale\n")

    data_loader = MemoryEfficientDataLoader(
        args.data_dir, args.seq_len, args.batch_size, world_size, rank
    )

    total_loss = 0.0
    log_loss = 0.0
    checkpoint_start_time = time.time()
    checkpoint_count = 0

    for step, (input_ids, labels) in enumerate(data_loader):
        if step < start_step:
            continue

        current_lr = get_lr(scheduler_step, args.warmup_steps, args.total_steps, args.peak_lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, 50304), labels.view(-1), reduction="mean")

        scaler.scale(loss / args.gradient_accumulation_steps).backward()

        total_loss += loss.item()
        log_loss += loss.item()
        tokens_seen += input_ids.numel() * world_size

        if (step + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler_step += 1

        if (step + 1) % args.log_every == 0 and rank == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_seen / elapsed
            mfu = (tokens_seen * 50304 * 2) / (elapsed * 125e12 * world_size)
            avg_loss = log_loss / args.log_every
            grad_scale = scaler.get_scale()

            print(
                f"Step {step+1} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                f"Tokens/s: {tokens_per_sec:.0f} | MFU: {mfu:.2%} | Grad Scale: {grad_scale:.0f}"
            )

            if writer:
                writer.add_scalar("train/loss", avg_loss, step)
                writer.add_scalar("train/lr", current_lr, step)
                writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step)
                writer.add_scalar("train/mfu", mfu, step)
                writer.add_scalar("train/grad_scale", grad_scale, step)

            with open(csv_path, "a") as f:
                f.write(f"{step},{avg_loss:.6f},{current_lr:.2e},{tokens_per_sec:.1f},{mfu:.4f},{grad_scale:.0f}\n")

            log_loss = 0.0

        checkpoint_elapsed = (time.time() - checkpoint_start_time) / 60
        if checkpoint_elapsed >= args.checkpoint_every and rank == 0:
            ckpt_path = output_dir / f"ckpt_step_{step}.pt"
            torch.save({
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "scheduler_step": scheduler_step,
                "tokens_seen": tokens_seen,
                "seed": torch.seed(),
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
            checkpoint_start_time = time.time()
            checkpoint_count += 1

            old_ckpts = sorted((output_dir / "ckpt_*.pt").glob("ckpt_*.pt"))
            while len(old_ckpts) > args.max_checkpoints:
                old_ckpts[0].unlink()
                old_ckpts.pop(0)

        if scheduler_step >= args.total_steps:
            if rank == 0:
                print(f"Training complete at step {step}")
            break

    if rank == 0 and writer:
        writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()
