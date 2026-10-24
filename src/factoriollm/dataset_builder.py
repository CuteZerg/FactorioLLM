"""
Dataset builder (v4 pipeline orchestrator).

    scrape     download new blueprints from factorioprints.com into the `blueprints` table
    canonical  local: IR -> filters -> macro-lifted code -> sandbox verification (canonical_builder.py)
    prompts    Gemini: grounded EN/RU prompts from blueprint facts (prompt_generator.py)
    export     JSONL for training with rotation augmentation and train/eval split (export_dataset.py)

Usage: uv run python -m factoriollm.dataset_builder <stage> [options]
"""

from __future__ import annotations

import argparse
import asyncio
import os

from factoriollm.canonical_builder import build_canonical
from factoriollm.export_dataset import export_dataset_to_jsonl
from factoriollm.prompt_generator import generate_prompts
from factoriollm.scraper import scrape_and_filter


def main():
    parser = argparse.ArgumentParser(description="FactorioLLM dataset pipeline (v4)")
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("scrape", help="Download new blueprints into PostgreSQL")
    p.add_argument("--batch", type=int, default=None, help="Number of new blueprints to download (default: all)")

    p = sub.add_parser("canonical", help="Build verified canonical scripts (local, no LLM)")
    p.add_argument("--min-favorites", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verify", choices=["sandbox", "local"], default="sandbox")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rebuild", action="store_true")

    p = sub.add_parser("prompts", help="Generate grounded prompts with Gemini (free-tier friendly)")
    p.add_argument("--model", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--rpm", type=float, default=float(os.getenv("GEMINI_RPM", "8")))
    p.add_argument("--max-requests", type=int, default=int(os.getenv("GEMINI_MAX_REQUESTS", "100")))
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("export", help="Export dataset_v4.jsonl (+ eval split)")
    p.add_argument("--eval-fraction", type=float, default=0.03)
    p.add_argument("--no-rotate", action="store_true")
    p.add_argument("--max-seq-length", type=int, default=4096)

    args = parser.parse_args()
    if args.stage == "scrape":
        asyncio.run(scrape_and_filter(batch_size=args.batch))
    elif args.stage == "canonical":
        asyncio.run(build_canonical(
            min_favorites=args.min_favorites, limit=args.limit, verify=args.verify,
            workers=args.workers, rebuild=args.rebuild,
        ))
    elif args.stage == "prompts":
        asyncio.run(generate_prompts(args.model, args.batch_size, args.rpm, args.max_requests, args.limit))
    elif args.stage == "export":
        asyncio.run(export_dataset_to_jsonl(
            eval_fraction=args.eval_fraction, rotate=not args.no_rotate, max_seq_length=args.max_seq_length,
        ))


if __name__ == "__main__":
    main()
