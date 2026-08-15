"""
Treino distribuído (DDP) do CustomLM

Uso (1 nó, exemplo com 6 GPUs):
    torchrun --standalone --nproc_per_node=6 train.py --data_dir data/meucorpus
"""

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import CustomLM
from data import BinDataset


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--vocab_size", type=int, default=None,
                     help="normalmente não precisa passar, é lido de data_dir/meta.json. "
                          "Só usa isso pra sobrescrever manualmente (dados antigos sem meta.json).")

    # arquitetura
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--n_layers", type=int, default=16)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--conv_kernel", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=64)
    ap.add_argument("--mlp_mult", type=float, default=4.0)

    # treino
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--micro_batch_size", type=int, default=16, help="batch por GPU por passo")
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--ckpt_interval", type=int, default=1000)
    ap.add_argument("--grad_checkpointing", action="store_true")
    return ap.parse_args()


def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def get_lr(step, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    if step > args.max_steps:
        return args.min_lr
    decay_ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


@torch.no_grad()
def estimate_loss(raw_model, dataset_val, args, device):
    raw_model.eval()
    losses = []
    for _ in range(args.eval_iters):
        x, y = dataset_val.get_batch(args.micro_batch_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss, _ = raw_model(x, targets=y)
        losses.append(loss.item())
    raw_model.train()
    return sum(losses) / len(losses)


def main():
    args = get_args()
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_master = rank == 0

    if args.vocab_size is None:
        meta_path = os.path.join(args.data_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                args.vocab_size = json.load(f)["vocab_size"]
            if is_master:
                print(f"vocab_size lido de {meta_path}: {args.vocab_size:,}")
        else:
            args.vocab_size = 256  # fallback: dados antigos sem meta.json (byte-level)
            if is_master:
                print(f"meta.json não encontrado em {args.data_dir}, assumindo vocab_size=256 (byte-level)")

    if is_master:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"world_size={world_size} | device={device}")

    model = CustomLM(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        conv_kernel=args.conv_kernel,
        chunk_size=args.chunk_size,
        mlp_mult=args.mlp_mult,
    ).to(device)

    if args.grad_checkpointing:
        # economiza memória trocando por mais recomputação — útil se quiser
        # aumentar batch/contexto além do que 48GB aguenta direto
        model._grad_checkpointing = True

    if is_master:
        print(f"parametros do modelo: {model.num_params() / 1e6:.1f}M")

    raw_model = model
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    dataset_train = BinDataset(args.data_dir, "train", args.block_size)
    dataset_val = BinDataset(args.data_dir, "val", args.block_size)

    model.train()
    t0 = time.time()
    for step in range(args.max_steps):
        lr = get_lr(step, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro_step in range(args.grad_accum_steps):
            x, y = dataset_train.get_batch(args.micro_batch_size, device)
            # só sincroniza gradientes no último micro-step (padrão DDP + grad accum)
            model.require_backward_grad_sync = (micro_step == args.grad_accum_steps - 1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, targets=y)
                loss = loss / args.grad_accum_steps
            loss.backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if is_master and step % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_per_step = args.micro_batch_size * args.grad_accum_steps * args.block_size * world_size
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                  f"{tok_per_step / dt:,.0f} tok/s | {dt*1000/args.log_interval:.1f} ms/it")

        if step % args.eval_interval == 0 and step > 0:
            val_loss = estimate_loss(raw_model, dataset_val, args, device)
            if is_master:
                print(f"  [eval] step {step} | val_loss {val_loss:.4f}")

        if is_master and step % args.ckpt_interval == 0 and step > 0:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_{step}.pt")
            torch.save({
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "args": vars(args),
            }, ckpt_path)
            print(f"  [ckpt] salvo em {ckpt_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
