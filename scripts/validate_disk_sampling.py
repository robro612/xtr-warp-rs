#!/usr/bin/env python3
"""One-off validator for DiskSource sampling implementations.

Creates synthetic chunked embedding files on disk, then compares:
1) correctness: serial vs parallel sampled tensors
2) runtime: rough wall-clock timings for both implementations
"""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from xtr_warp.search import DiskSource


def _write_synthetic_embeddings(
    out_dir: Path,
    num_files: int = 12,
    docs_per_file: int = 400,
    dim: int = 128,
    min_len: int = 24,
    max_len: int = 128,
    seed: int = 13,
) -> tuple[int, int]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    total_docs = 0
    total_tokens = 0

    for file_idx in range(num_files):
        doclens = [rng.randint(min_len, max_len) for _ in range(docs_per_file)]
        n_tokens = int(sum(doclens))
        data = np_rng.standard_normal((n_tokens, dim), dtype=np.float32)

        emb_path = out_dir / f"emb_{file_idx}.npy"
        doclens_path = out_dir / f"emb_{file_idx}.doclens.npy"
        np.save(emb_path, data)
        np.save(doclens_path, np.asarray(doclens, dtype=np.int64))

        total_docs += docs_per_file
        total_tokens += n_tokens

    return total_docs, total_tokens


def _time_call(fn, rounds: int = 3) -> tuple[tuple[torch.Tensor, int, int], float]:
    out = None
    start = time.perf_counter()
    for _ in range(rounds):
        out = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / rounds
    return out, elapsed_ms


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DiskSource sampling implementations.")
    parser.add_argument("--num-files", type=int, default=12)
    parser.add_argument("--docs-per-file", type=int, default=400)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--min-len", type=int, default=24)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--sample-size", type=int, default=1800)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--data-seed", type=int, default=13)
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tmp_root = Path(tempfile.mkdtemp(prefix="xtr_warp_disk_validate_"))
    print(f"[validate] temp dir: {tmp_root}")
    try:
        total_docs, total_tokens = _write_synthetic_embeddings(
            tmp_root,
            num_files=args.num_files,
            docs_per_file=args.docs_per_file,
            dim=args.dim,
            min_len=args.min_len,
            max_len=args.max_len,
            seed=args.data_seed,
        )
        print(f"[validate] dataset: docs={total_docs}, tokens={total_tokens}")

        source = DiskSource(tmp_root)
        num_passages = source.get_num_passages()
        assert num_passages == total_docs

        sample_size = min(args.sample_size, num_passages)
        sampled_pids = random.Random(args.sample_seed).sample(range(num_passages), k=sample_size)
        print(f"[validate] sampling {sample_size} pids")

        (serial_t, serial_tokens, serial_dim), serial_ms = _time_call(
            lambda: source.sample_embeddings_serial(sampled_pids),
            rounds=args.rounds,
        )
        (parallel_t, parallel_tokens, parallel_dim), parallel_ms = _time_call(
            lambda: source.sample_embeddings_parallel(sampled_pids),
            rounds=args.rounds,
        )

        assert serial_tokens == parallel_tokens
        assert serial_dim == parallel_dim
        assert serial_t.shape == parallel_t.shape
        assert torch.equal(serial_t, parallel_t), "Serial and parallel tensors differ"

        print("[validate] correctness: PASS (serial == parallel)")
        print(f"[validate] serial avg:   {serial_ms:.2f} ms")
        print(f"[validate] parallel avg: {parallel_ms:.2f} ms")
        if parallel_ms > 0:
            print(f"[validate] speedup:      {serial_ms / parallel_ms:.2f}x")
    finally:
        if args.keep_temp:
            print(f"[validate] keeping temp dir: {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
