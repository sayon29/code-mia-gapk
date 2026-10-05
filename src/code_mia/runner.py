from __future__ import annotations

import csv
import gc
import json
import shutil
import time
from pathlib import Path

import torch
import yaml

from .attacks.gapk import score_from_logits
from .config import config_fingerprint, dump_yaml
from .data.manifest import build_manifest, read_jsonl, write_json_atomic, write_jsonl_atomic
from .evaluation.metrics import calculate_metrics
from .evaluation.plots import plot_histogram, plot_roc, write_summary_csv
from .models.starcoder2 import load_model_and_tokenizer
from .utils.environment import collect_environment, estimate_memory
from .utils.logging import configure_logging
from .utils.reproducibility import set_deterministic_seed


SCORE_FIELDS = [
    "sample_id", "membership", "canonical_text_sha256", "starcoder_token_count",
    "raw_gap_score_count", "smoothed_score_count", "selected_bottom_count", "gapk_score",
    "model_id", "model_revision", "tokenizer_id", "tokenizer_revision", "model_dtype",
    "statistics_dtype", "gap_fraction", "smoothing_window", "quantization", "config_fingerprint",
    "runtime_seconds", "peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes", "status", "error_message",
]


def prepare_run(config: dict, original_config_path: Path) -> tuple[Path, str]:
    run_dir = Path(config["experiment"]["output_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original_config_path, run_dir / "config.original.yaml")
    dump_yaml(config, run_dir / "config.resolved.yaml")
    fp = config_fingerprint(config)
    (run_dir / "config.fingerprint").write_text(fp + "\n", encoding="utf-8")
    write_json_atomic(collect_environment(config["model"]["id"], config["dataset"]["id"]), run_dir / "environment.json")
    return run_dir, fp


def _compatibility(config: dict, fp: str, row: dict, dtype_name: str) -> dict:
    return {
        "sample_id": row["sample_id"], "canonical_text_sha256": row["canonical_text_sha256"],
        "model_id": config["model"]["id"], "model_revision": config["model"]["revision"],
        "tokenizer_id": config["model"]["tokenizer_id"], "tokenizer_revision": config["model"]["tokenizer_revision"],
        "model_dtype": dtype_name, "statistics_dtype": "float32",
        "gap_fraction": float(config["attack"]["gap_fraction"]),
        "smoothing_window": int(config["attack"]["smoothing_window"]),
        "quantization": config["model"].get("quantization", "none"), "config_fingerprint": fp,
    }


def _compatible(cached: dict, expected: dict) -> bool:
    return all(cached.get(key) == value for key, value in expected.items()) and cached.get("status") == "success"


def attack(config: dict, run_dir: Path, fp: str) -> list[dict]:
    logger = configure_logging(run_dir / "run.log")
    manifest = read_jsonl(run_dir / "manifest.jsonl")
    if not manifest:
        raise RuntimeError("Frozen manifest is absent or empty; run build-manifest first")
    if not torch.cuda.is_available() and not config["runtime"].get("allow_cpu_attack", False):
        raise RuntimeError("CUDA is unavailable. Manifest/tests remain usable; set runtime.allow_cpu_attack=true only for an explicit slow CPU run.")
    memory = estimate_memory(config, max(row["starcoder_token_count"] for row in manifest))
    logger.info("Pre-load memory estimate: %s", json.dumps(memory, sort_keys=True))
    loaded = load_model_and_tokenizer(config["model"])
    scores_path = run_dir / "scores.jsonl"
    cached_rows = {row["sample_id"]: row for row in read_jsonl(scores_path)}
    results: list[dict] = []
    total_start = time.perf_counter()
    for ordinal, row in enumerate(manifest, 1):
        expected = _compatibility(config, fp, row, loaded.dtype_name)
        cached = cached_rows.get(row["sample_id"])
        if cached and _compatible(cached, expected):
            results.append(cached); logger.info("[%d/%d] resume %s", ordinal, len(manifest), row["sample_id"][:12]); continue
        start = time.perf_counter()
        encoded = input_ids = logits = None
        base = {**expected, "membership": row["membership"], "starcoder_token_count": row["starcoder_token_count"]}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(loaded.device)
        try:
            encoded = loaded.tokenizer(row["canonical_text"], return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"].to(loaded.device)
            with torch.inference_mode():
                logits = loaded.model(input_ids=input_ids).logits
                result = score_from_logits(logits, input_ids, float(config["attack"]["gap_fraction"]),
                                           int(config["attack"]["smoothing_window"]),
                                           float(config["attack"].get("variance_floor", 1e-8)))
            score_row = {**base, "raw_gap_score_count": result.raw_count,
                         "smoothed_score_count": result.smoothed_count, "selected_bottom_count": result.selected_count,
                         "gapk_score": result.score, "runtime_seconds": time.perf_counter() - start,
                         "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(loaded.device)) if torch.cuda.is_available() else 0,
                         "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(loaded.device)) if torch.cuda.is_available() else 0,
                         "status": "success", "error_message": ""}
        except Exception as exc:
            score_row = {**base, "raw_gap_score_count": None, "smoothed_score_count": None,
                         "selected_bottom_count": None, "gapk_score": None, "runtime_seconds": time.perf_counter() - start,
                         "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(loaded.device)) if torch.cuda.is_available() else 0,
                         "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(loaded.device)) if torch.cuda.is_available() else 0,
                         "status": "failure", "error_message": f"{type(exc).__name__}: {exc}"}
            logger.exception("[%d/%d] failed %s", ordinal, len(manifest), row["sample_id"][:12])
        results.append(score_row)
        write_jsonl_atomic(results, scores_path)
        del encoded, input_ids, logits
        gc.collect()
        interval = int(config["runtime"].get("cuda_cache_cleanup_interval", 0))
        if interval and torch.cuda.is_available() and ordinal % interval == 0: torch.cuda.empty_cache()
        logger.info("[%d/%d] %s %.3fs", ordinal, len(manifest), score_row["status"], score_row["runtime_seconds"])
    write_json_atomic({"total_runtime_seconds": time.perf_counter() - total_start}, run_dir / "runtime.json")
    _write_scores_csv(results, run_dir / "scores.csv")
    write_jsonl_atomic([row for row in results if row["status"] != "success"], run_dir / "failures.jsonl")
    return results


def _write_scores_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def evaluate(run_dir: Path) -> dict:
    rows = read_jsonl(run_dir / "scores.jsonl")
    successes = [row for row in rows if row.get("status") == "success"]
    failures = len(rows) - len(successes) + len(read_jsonl(run_dir / "exclusions.jsonl"))
    metrics, curve = calculate_metrics([row["membership"] for row in successes], [row["gapk_score"] for row in successes], failures)
    write_json_atomic(metrics, run_dir / "metrics.json")
    lines = ["# Gap-K% metrics", "", f"- ROC-AUC: {metrics['roc_auc']:.6f}",
             f"- TPR @ 5% FPR: {metrics['tpr_at_fpr']['0.05']:.6f}",
             f"- TPR @ 1% FPR: {metrics['tpr_at_fpr']['0.01']:.6f}",
             f"- TPR @ 0.1% FPR: {metrics['tpr_at_fpr']['0.001']:.6f}",
             f"- Members / non-members: {metrics['member_count']} / {metrics['non_member_count']}",
             f"- Successful / failed or excluded: {metrics['successful_score_count']} / {metrics['failure_exclusion_count']}",
             f"- Minimum nonzero empirical FPR: {metrics['minimum_nonzero_empirical_fpr']}",
             f"- Direction: {metrics['score_direction']}"]
    if metrics["warnings"]: lines += ["", "## Warnings", ""] + [f"- {warning}" for warning in metrics["warnings"]]
    (run_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_roc(curve, run_dir / "roc_curve.png")
    plot_histogram([row["membership"] for row in successes], [row["gapk_score"] for row in successes], run_dir / "score_histogram.png")
    write_summary_csv(successes, run_dir / "score_summary.csv")
    return metrics


def validate_run(run_dir: Path) -> list[str]:
    required = ["config.original.yaml", "config.resolved.yaml", "config.fingerprint", "environment.json",
                "manifest.jsonl", "exclusions.jsonl", "dataset_audit.json", "scores.csv", "failures.jsonl",
                "metrics.json", "metrics.md", "roc_curve.png", "score_histogram.png", "run.log"]
    errors = [f"missing {name}" for name in required if not (run_dir / name).exists()]
    if (run_dir / "scores.jsonl").exists() and (run_dir / "manifest.jsonl").exists():
        manifest_ids = {row["sample_id"] for row in read_jsonl(run_dir / "manifest.jsonl")}
        score_rows = read_jsonl(run_dir / "scores.jsonl")
        if {row["sample_id"] for row in score_rows} != manifest_ids: errors.append("score sample IDs do not exactly match manifest IDs")
    return errors
