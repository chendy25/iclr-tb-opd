#!/usr/bin/env python3
"""Offline stub parquet for ATOD / verl-agent ALFWorld training.

ATOD's examples/data_preprocess/prepare.py downloads hiyouga/geometry3k only to
size the dataset; ALFWorld games come from $ALFWORLD_DATA. This script writes
the same schema without any HuggingFace download — safe on air-gapped jobs.
"""
from __future__ import annotations

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq


def _rows(split: str, n: int) -> dict:
    return {
        "data_source": ["text"] * n,
        "prompt": [[{"role": "user", "content": ""}] for _ in range(n)],
        "ability": ["agent"] * n,
        "extra_info": [{"split": split, "index": i} for i in range(n)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", required=True, help="Output dir (writes train/test.parquet)")
    ap.add_argument("--train_data_size", type=int, default=16)
    ap.add_argument("--val_data_size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    train_path = os.path.join(args.local_dir, "train.parquet")
    test_path = os.path.join(args.local_dir, "test.parquet")

    pq.write_table(pa.table(_rows("train", args.train_data_size)), train_path)
    pq.write_table(pa.table(_rows("test", args.val_data_size)), test_path)
    print(f"[prepare_text_stubs] wrote {train_path} ({args.train_data_size} rows)")
    print(f"[prepare_text_stubs] wrote {test_path} ({args.val_data_size} rows)")


if __name__ == "__main__":
    main()
