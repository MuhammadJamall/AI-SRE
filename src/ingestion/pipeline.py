"""Usage (paths default to configs/default.yaml):
python -m src.ingestion.pipeline smd
python -m src.ingestion.pipeline hdfs [--max-lines 500000]
"""
import argparse
import json
from pathlib import Path

from src.common.config import load_config, resolve
from src.common.logging_utils import setup_logging
from src.ingestion.hdfs import ingest_hdfs
from src.ingestion.smd import ingest_smd


def main(argv=None):
    p = argparse.ArgumentParser(description="Aether Module 1: ingestion")
    p.add_argument("dataset", choices=["smd", "hdfs"])
    p.add_argument("--config", default=None)
    p.add_argument("--raw", default=None, help="override raw dataset path")
    p.add_argument("--out", default=None, help="override output directory")
    p.add_argument("--max-lines", type=int, default=None, help="HDFS only: limit lines (quick test)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    raw = resolve(args.raw or cfg["datasets"][args.dataset]["raw_path"])
    out = resolve(args.out or Path(cfg["data"]["processed_dir"]) / args.dataset)

    if args.dataset == "smd":
        ingest_smd(raw, out)
    else:
        h = cfg["datasets"]["hdfs"]
        ingest_hdfs(raw, out, max_lines=args.max_lines,
                    batch_size=h.get("batch_size", 200_000), train_ratio=h.get("train_ratio", 0.7))

    print(json.dumps(json.loads((out / "report.json").read_text()), indent=2))


if __name__ == "__main__":
    main()