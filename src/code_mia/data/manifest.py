from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable

from ..utils.hashing import fingerprint, sha256_text
from .audit import build_audit
from .canonicalize import canonicalize_prefix, word_spans
from .poisoned_chalice import load_public_test, membership_label


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_jsonl_atomic(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def manifest_protocol(config: dict) -> dict:
    return {
        "dataset": {key: config["dataset"].get(key) for key in ("id", "revision", "config", "split")},
        "counts": {
            "members": config["dataset"]["members"],
            "non_members": config["dataset"]["non_members"],
        },
        "seed": config["experiment"]["seed"],
        "word_units": config["canonicalization"]["word_units"],
        "tokenizer": {
            "id": config["model"]["tokenizer_id"],
            "revision": config["model"]["tokenizer_revision"],
        },
    }


def deterministic_stratified_indices(rows: list[dict], members: int, non_members: int, seed: int) -> list[int]:
    by_class = {0: [], 1: []}
    for index, row in enumerate(rows):
        if row.get("eligibility_status") == "eligible":
            by_class[int(row["membership"])].append(index)
    rng = random.Random(seed)
    for indices in by_class.values():
        indices.sort(key=lambda i: rows[i]["sample_id"])
        rng.shuffle(indices)
    if len(by_class[1]) < members or len(by_class[0]) < non_members:
        raise ValueError(
            f"Insufficient eligible rows: members={len(by_class[1])}/{members}, "
            f"non-members={len(by_class[0])}/{non_members}"
        )
    return by_class[1][:members] + by_class[0][:non_members]


def build_manifest(config: dict, run_dir: Path, tokenizer=None, force: bool = False) -> tuple[list[dict], dict]:
    manifest_path = run_dir / "manifest.jsonl"
    metadata_path = run_dir / "manifest.metadata.json"
    protocol = manifest_protocol(config)
    protocol_fingerprint = fingerprint(protocol)
    if manifest_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("protocol_fingerprint") != protocol_fingerprint:
            raise RuntimeError("Frozen manifest exists but its protocol differs; choose a new output directory or --force")
        rows = read_jsonl(manifest_path)
        expected = config["dataset"]["members"] + config["dataset"]["non_members"]
        if len(rows) != expected:
            raise RuntimeError(f"Frozen manifest is incomplete: expected {expected}, found {len(rows)}")
        return rows, metadata

    if tokenizer is None:
        from ..models.starcoder2 import load_tokenizer
        tokenizer = load_tokenizer(config["model"])
    dataset = load_public_test(config["dataset"])
    text_column = config["dataset"].get("text_column", "content")
    label_column = config["dataset"].get("label_column", "membership")
    if text_column not in dataset.column_names or label_column not in dataset.column_names:
        raise KeyError(f"Dataset columns are {dataset.column_names}; expected {text_column!r} and {label_column!r}")

    eligible: list[dict] = []
    exclusions: list[dict] = []
    units = int(config["canonicalization"]["word_units"])
    ds = config["dataset"]
    for row_index, source in enumerate(dataset):
        raw = source[text_column]
        try:
            label = membership_label(source[label_column])
        except ValueError as exc:
            exclusions.append({"original_row_index": row_index, "exclusion_reason": "invalid_membership", "detail": str(exc)})
            continue
        canonical = canonicalize_prefix(raw, units)
        raw_string = raw if isinstance(raw, str) else ""
        raw_hash = sha256_text(raw_string)
        identity = {
            "dataset_name": ds["id"], "dataset_revision": ds["revision"],
            "language_configuration": ds["config"], "split": ds["split"],
            "original_row_index": row_index, "raw_text_sha256": raw_hash,
        }
        sample_id = fingerprint(identity)
        if not canonical.eligible:
            exclusions.append({
                "sample_id": sample_id, **identity, "membership": label,
                "eligibility_status": "excluded", "exclusion_reason": canonical.exclusion_reason,
            })
            continue
        encoded = tokenizer(canonical.text, add_special_tokens=True, return_attention_mask=False)
        record = {
            "sample_id": sample_id, **identity, "membership": label,
            "raw_text": raw_string, "canonical_text": canonical.text,
            "canonical_text_sha256": sha256_text(canonical.text),
            "raw_byte_count": len(raw_string.encode("utf-8")),
            "raw_character_count": len(raw_string),
            "raw_line_count": len(raw_string.splitlines()),
            "nltk_word_count": canonical.word_count,
            "canonical_word_count": len(word_spans(canonical.text)),
            "starcoder_token_count": len(encoded["input_ids"]),
            "eligibility_status": "eligible", "exclusion_reason": None,
        }
        eligible.append(record)

    selected_indices = deterministic_stratified_indices(
        eligible, int(ds["members"]), int(ds["non_members"]), int(config["experiment"]["seed"])
    )
    selected = [eligible[index] for index in selected_indices]
    audit = build_audit(selected, exclusions)
    metadata = {"protocol": protocol, "protocol_fingerprint": protocol_fingerprint, "audit": audit}
    write_jsonl_atomic(selected, manifest_path)
    write_jsonl_atomic(exclusions, run_dir / "exclusions.jsonl")
    write_json_atomic(audit, run_dir / "dataset_audit.json")
    write_json_atomic(metadata, metadata_path)
    return selected, metadata

