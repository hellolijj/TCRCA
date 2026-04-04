#!/usr/bin/env python3
"""
Aggregate evaluation from a JSONL manifest (root cause + optional path / alert labels).

Example:
    python experiments/run_evaluation.py \\
        --manifest experiments/eval_manifest.example.jsonl \\
        --dataset-root datasets/train_ticket \\
        --epochs 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcrca.config import TCRCAConfig
from tcrca.data import AIOpsLiveBenchmarkDataSource, DataSourceManager, TrainTicketDataSource
from tcrca.eval.runner import evaluate_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="TCRCA offline evaluation (manifest-driven).")
    ap.add_argument("--manifest", type=Path, required=True, help="JSONL file; see experiments/eval_manifest.example.jsonl")
    ap.add_argument(
        "--dataset",
        choices=("train_ticket", "aiops_live"),
        default="train_ticket",
        help="Benchmark per Sec. V (Train Ticket controlled testbed vs. AIOps Live).",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override root folder (default: datasets/train_ticket or datasets/aiops_live).",
    )
    ap.add_argument("--epochs", type=int, default=40, help="Unsupervised training steps per case")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-structure", type=float, default=None, help="Override TCRCAConfig.lambda_structure")
    args = ap.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    cfg = TCRCAConfig()
    if args.lambda_structure is not None:
        cfg.lambda_structure = args.lambda_structure

    if args.dataset_root is not None:
        root = args.dataset_root
    elif args.dataset == "train_ticket":
        root = ROOT / "datasets" / "train_ticket"
    else:
        root = ROOT / "datasets" / "aiops_live"

    manager = DataSourceManager()
    if args.dataset == "train_ticket":
        manager.add_data_source(TrainTicketDataSource(str(root)))
    else:
        manager.add_data_source(AIOpsLiveBenchmarkDataSource(str(root)))
    data = manager.get_data()

    for key in ("traces", "metrics", "logs"):
        if key not in data or data[key] is None or data[key].empty:
            logger.warning("Table '%s' is empty — evaluation may fail.", key)

    records, summary = evaluate_manifest(args.manifest, data, cfg=cfg, train_epochs=args.epochs, lr=args.lr)

    print(json.dumps(summary, indent=2))
    per_path = args.manifest.with_suffix(".per_case.jsonl")
    with open(per_path, "w", encoding="utf-8") as out:
        for r in records:
            out.write(
                json.dumps(
                    {
                        "case_id": r.case_id,
                        "A@1": r.hit_at_1,
                        "A@3": r.hit_at_3,
                        "A@5": r.hit_at_5,
                        "hit@1": r.hit_at_1,
                        "hit@3": r.hit_at_3,
                        "hit@5": r.hit_at_5,
                        "rank": r.rank,
                        "rr": r.reciprocal_rank,
                        "edge_jaccard": r.edge_jaccard,
                        "edge_f1": r.edge_f1,
                        "extras": r.extras,
                    }
                )
                + "\n"
            )
    logger.info("Wrote per-case metrics to %s", per_path)


if __name__ == "__main__":
    main()
