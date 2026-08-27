"""Two-rank, 1 GiB NCCL all-reduce smoke test for the Podman image."""

from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-mib", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser.parse_args()


def emit(rank: int, stage: str, **details: object) -> None:
    fields = ["NCCL_SMOKE", f"rank={rank}", f"stage={stage}"]
    fields.extend(f"{key}={value}" for key, value in details.items())
    print(" ".join(fields), flush=True)


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"NCCL smoke test requires WORLD_SIZE=2, got {world_size}")
    if args.tensor_mib <= 0 or args.iterations <= 0:
        raise ValueError("--tensor-mib and --iterations must be positive")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    init_started = time.monotonic()
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=args.timeout_seconds),
        device_id=device,
    )
    try:
        emit(
            rank,
            "initialized",
            elapsed_seconds=f"{time.monotonic() - init_started:.3f}",
            device=torch.cuda.get_device_name(local_rank).replace(" ", "_"),
        )
        peer_rank = 1 - local_rank
        emit(
            rank,
            "peer_access",
            peer_local_rank=peer_rank,
            enabled=torch.cuda.can_device_access_peer(local_rank, peer_rank),
        )

        float32_bytes = torch.empty((), dtype=torch.float32).element_size()
        element_count = args.tensor_mib * 2**20 // float32_bytes
        tensor = torch.full(
            (element_count,),
            float(rank + 1),
            dtype=torch.float32,
            device=device,
        )
        expected = world_size * (world_size + 1) / 2
        dist.barrier(device_ids=[local_rank])
        all_reduce_started = time.monotonic()
        for _ in range(args.iterations):
            tensor.fill_(float(rank + 1))
            dist.all_reduce(tensor)
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - all_reduce_started
        actual = tensor[0].item()
        if actual != expected:
            raise RuntimeError(f"all-reduce result {actual}, expected {expected}")
        emit(
            rank,
            "NCCL_SMOKE_PASS",
            iterations=args.iterations,
            tensor_mib=args.tensor_mib,
            elapsed_seconds=f"{elapsed:.3f}",
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
