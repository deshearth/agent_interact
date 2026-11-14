"""Reproducible command entry point; test evaluation is explicitly gated."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect-environment")
    p = sub.add_parser("prepare-data")
    p.add_argument("--config", default="configs/paper_binary.yaml")
    p.add_argument("--download", action="store_true")
    p.add_argument("--raw", action="store_true")
    p = sub.add_parser("audit-data")
    p.add_argument("--config", default="configs/paper_binary.yaml")
    p = sub.add_parser("encode-messages")
    p.add_argument("--config", default="configs/paper_binary.yaml")
    p.add_argument("--partition", choices=["train", "validation", "development", "all"], default="train")
    p.add_argument("--limit-episodes", type=int)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("train")
    p.add_argument("--config", default="configs/paper_binary.yaml")
    p.add_argument("--model", choices=["P0", "P1", "I0", "A0", "A1", "A2"], default=None)
    p.add_argument("--state", choices=["opinion", "vote_counts", "history", "existing_messages"])
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--wall-minutes", type=float)
    p.add_argument("--limit-episodes", type=int)
    p.add_argument("--match-state", choices=["existing_messages"])
    p.add_argument("--resume", type=Path)
    p.add_argument("--smoke", action="store_true", help="Mark a preparation run ineligible for final selection")
    p = sub.add_parser("reproduce-baseline")
    p.add_argument("--config", default="configs/paper_binary.yaml")
    p = sub.add_parser("freeze")
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("runs/selection.json"))
    p.add_argument("--rationale", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--partition", choices=["validation", "test"], default="validation")
    p.add_argument("--mode", choices=["one_step", "closed_loop"], default="one_step")
    p.add_argument("--selection", type=Path)
    p.add_argument("--rollouts", type=int)
    p = sub.add_parser("report")
    p.add_argument("--experiment", default="paper_binary")
    p = sub.add_parser("screen")
    p.add_argument("--config", default="configs/paper_binary.yaml")
    p.add_argument("--minutes", type=float, default=120)
    p.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.command == "inspect-environment":
        from .common import inspect_environment
        inspect_environment()
    elif args.command in ("prepare-data", "audit-data"):
        from .common import config
        from .data.prepare import prepare
        cfg = config(args.config)
        if getattr(args, "download", False):
            from .data.acquire import acquire
            acquire(Path(cfg["data"]["root"]), raw=args.raw)
        prepare(cfg)
    elif args.command == "encode-messages":
        from .common import config
        from .data.messages import encode_messages
        encode_messages(config(args.config), args.partition, args.limit_episodes, args.device)
    elif args.command == "train":
        from .train import train
        train(args)
    elif args.command == "reproduce-baseline":
        from .common import config
        from .models.baselines import reproduce
        reproduce(config(args.config))
    elif args.command == "freeze":
        from .evaluate import freeze
        freeze(args.runs, args.output, args.rationale)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(args)
    elif args.command == "report":
        from .report import report
        report(args.experiment)
    elif args.command == "screen":
        from .screen import screen
        screen(args.config, args.minutes, args.device)


if __name__ == "__main__":
    main()
