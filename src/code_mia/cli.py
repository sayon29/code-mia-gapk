from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .config import config_fingerprint, load_config
from .data.manifest import build_manifest
from .runner import attack, evaluate, prepare_run, validate_run
from .utils.environment import collect_environment, print_environment
from .utils.reproducibility import set_deterministic_seed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="code-mia", description="Vanilla Gap-K% baseline for StarCoder2-3B")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect-environment", help="report CUDA, packages, memory, disk, network and caches")
    inspect.add_argument("--output", type=Path)
    for name in ("download", "build-manifest", "attack", "run"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True, type=Path)
        item.add_argument("--output-dir", type=Path)
        item.add_argument("--model-path", type=Path)
        item.add_argument("--dataset-path", type=Path)
        item.add_argument("--offline", action="store_true")
        if name == "build-manifest": item.add_argument("--force", action="store_true")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--run-dir", required=True, type=Path)
    validate = sub.add_parser("validate-run")
    validate.add_argument("--run-dir", required=True, type=Path)
    return parser


def _configured(args) -> dict:
    overrides = {
        "experiment.output_dir": str(args.output_dir) if args.output_dir else None,
        "model.local_path": str(args.model_path) if args.model_path else None,
        "dataset.local_path": str(args.dataset_path) if args.dataset_path else None,
    }
    if args.offline:
        overrides.update({"model.offline": True, "dataset.offline": True})
    return load_config(args.config, overrides)


def download_assets(config: dict) -> None:
    if config["model"].get("local_path") or config["dataset"].get("local_path"):
        print("Local asset path configured; validating through tokenizer/dataset loaders.")
    from .models.starcoder2 import load_tokenizer
    from .data.poisoned_chalice import load_public_test
    from huggingface_hub import snapshot_download
    load_tokenizer(config["model"])
    if not config["model"].get("local_path"):
        snapshot_download(config["model"]["id"], revision=config["model"]["revision"],
                          local_files_only=bool(config["model"].get("offline", False)))
    dataset = load_public_test(config["dataset"])
    print(f"Assets ready: tokenizer/model revision pinned; dataset rows={len(dataset)}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect-environment":
            report = collect_environment("bigcode/starcoder2-3b", "AISE-TUDelft/Poisoned-Chalice")
            print_environment(report)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return 0
        if args.command == "evaluate":
            metrics = evaluate(args.run_dir.resolve()); print(json.dumps(metrics, indent=2)); return 0
        if args.command == "validate-run":
            errors = validate_run(args.run_dir.resolve())
            if errors:
                print("Run validation failed:\n- " + "\n- ".join(errors), file=sys.stderr); return 1
            print(f"Run is valid: {args.run_dir.resolve()}"); return 0
        config = _configured(args)
        set_deterministic_seed(int(config["experiment"]["seed"]))
        if args.command == "download":
            download_assets(config); return 0
        run_dir, fp = prepare_run(config, args.config)
        if args.command in {"build-manifest", "run"}:
            rows, metadata = build_manifest(config, run_dir, force=getattr(args, "force", False))
            print(json.dumps(metadata["audit"], indent=2))
        if args.command in {"attack", "run"}:
            attack(config, run_dir, fp)
        if args.command == "run":
            metrics = evaluate(run_dir)
            errors = validate_run(run_dir)
            if errors: raise RuntimeError("Output validation failed: " + "; ".join(errors))
            print(json.dumps(metrics, indent=2)); print(f"Validated run: {run_dir}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
