# code-mia-gapk

An auditable, small-scale implementation of the **vanilla Gap-K%** membership-inference baseline for the original, unfine-tuned `bigcode/starcoder2-3b`, evaluated on the Python public `test` split of `AISE-TUDelft/Poisoned-Chalice`. The smoke run selects 20 members and 20 non-members; the debug run selects 100+100. Neither is a definitive scientific result.

Membership inference asks whether an example was likely present in a model's pretraining data. In Poisoned Chalice, `membership == 1` is treated as member and `membership == 0` as non-member. This repository evaluates the original pretrained target specifically: it does not fine-tune, add code-specific transformations, or run a larger model.

## Reproducibility boundaries

- Model and tokenizer: `bigcode/starcoder2-3b`, revision `733247c55e3f73af49ce8e9c7949bf14af205928`.
- Dataset: `AISE-TUDelft/Poisoned-Chalice`, revision `ae3f42011bf55ab2910c83a342caf8a3c02bcd5d`.
- Python public `test`, seed 42, exactly 32 NLTK Treebank word units.
- Gap fraction 0.20, valid moving-average window 3, batch size 1.
- Exact FP16/BF16 model inference with float32 Gap-K statistics; quantization is disabled.

Revision failures never fall back silently to latest. Explicit YAML/CLI local paths and revisions are recorded in resolved configuration. The window of 3 follows the extrapolated vanilla non-LLaMA/default-family setting; it was not tuned for StarCoder2.

## Method

For token `x_t`, logits at `t-1` predict that token; the first input token has no score. With full-vocabulary log probabilities `l_t(v)` and probabilities `p_t(v)`:

```text
mu_t      = sum_v p_t(v) l_t(v)
variance  = sum_v p_t(v) (l_t(v) - mu_t)^2
sigma_t   = sqrt(max(variance, 1e-8))
g_t       = (l_t(x_t) - max_v l_t(v)) / sigma_t
```

After a valid width-3 moving average, values are sorted ascending and the lowest `max(1, floor(0.20*n))` are averaged. Scores are normally non-positive. **Higher (closer to zero) is more member-like**, and label 1 is the ROC positive class. Full logits are essential because the probability-weighted vocabulary variance uses every token; a generation-only API or top-k response cannot reproduce it.

The implementation retains only the full float32 log-probability tensor and one reused full-vocabulary work buffer. No logits are saved. Each sample is checkpointed atomically with hashes, revisions, dtype, quantization mode, protocol parameters, and configuration fingerprint.

## Install and local checks

```bash
python -m pip install -e '.[test]'
code-mia --help
pytest -q
code-mia inspect-environment
```

CPU is supported for tests, environment inspection, and manifest work. An attack refuses to start on CPU unless `runtime.allow_cpu_attack: true` is explicitly set; a 3B CPU run is generally impractical.

## Frozen manifest

```bash
code-mia build-manifest --config configs/smoke.yaml
```

The builder retains the original row index, rejects null/empty/short content, and uses NLTK's `TreebankWordTokenizer.span_tokenize`, which is the span-producing equivalent of `word_tokenize(..., preserve_line=True)`. It cuts the raw string at the end of the 32nd word span—never joining tokens or normalizing source. The result is an exact original prefix. Eligible members and non-members are sampled separately and frozen. A matching manifest is reused without loading or resampling the dataset; a conflicting protocol is rejected.

`dataset_audit.json` reports counts, exclusions, lengths, and duplicate hashes. Raw/canonical text and hashes are retained for auditability; treat manifests as research data, not necessarily as safe public artifacts.

## Run and resume

```bash
code-mia run --config configs/smoke.yaml
code-mia run --config configs/debug_gapk_starcoder2_3b.yaml
code-mia validate-run --run-dir outputs/smoke
```

Re-run the same command after interruption. Successful compatible samples are reused; mismatched hashes, revisions, tokenizer, input protocol, K, smoothing, dtype, quantization, implementation/config fingerprint, or failed rows are recomputed. Quantized runs, if explicitly enabled (`4bit` or `8bit`, requiring bitsandbytes), are distinct experimental conditions and should use a separate output directory.

Metrics are ROC-AUC and empirical (non-interpolated) TPR at 5%, 1%, and 0.1% FPR. With 100 non-members, one false positive is 1%; 0.1% is unresolved. The JSON and Markdown explicitly warn that the reported 0.1% number is merely the best observed zero-FP point.

## Kaggle

1. Create a Kaggle notebook, enable a GPU under notebook settings, and enable Internet for the first download.
2. Upload this repository so it is available as `/kaggle/working/code-mia-gapk`, or set the real Git URL in the notebook clone cell.
3. Open `notebooks/kaggle_debug.ipynb` and run the setup, cache, install, environment, and test cells first.
4. Run the manifest cell, inspect its audit, then run the smoke cell. Run the separate 100+100 cell only when ready.

The notebook sets:

```text
HF_HOME=/kaggle/working/hf-cache
HUGGINGFACE_HUB_CACHE=/kaggle/working/hf-cache/hub
HF_DATASETS_CACHE=/kaggle/working/hf-cache/datasets
TOKENIZERS_PARALLELISM=false
```

It does not reinstall PyTorch explicitly. The loader prefers BF16 only when PyTorch reports genuine support; T4/P100 therefore use FP16. One GPU is used even if two are visible. Before loading, the run log records dtype, free VRAM, sequence length, approximate full-float32-logit size, and a conservative fit estimate. A StarCoder2-3B half-precision run typically needs roughly 7–10 GB peak VRAM depending on software/sequence length. After assets are cached, a 20+20 smoke run may take several minutes to tens of minutes; 100+100 may take roughly one to several hours depending on GPU and download/cache state. These are planning estimates, not measured results from this repository.

The final notebook cell writes `/kaggle/working/code-mia-gapk-results.zip`.

### Offline Kaggle reuse

With Internet enabled once, preserve `/kaggle/working/hf-cache`, or upload the cached model/tokenizer and a dataset `save_to_disk` snapshot as a private Kaggle Dataset. Mounted `/kaggle/input` is read-only; point `--model-path` and `--dataset-path` there and keep output under `/kaggle/working`:

```bash
code-mia run --config configs/smoke.yaml --offline \
  --model-path /kaggle/input/my-assets/starcoder2-3b \
  --dataset-path /kaggle/input/my-assets/poisoned-chalice-python \
  --output-dir /kaggle/working/code-mia-gapk/outputs/smoke-offline
```

The model directory must be a complete Transformers snapshot including `config.json`; the dataset path must be a `datasets.save_to_disk` snapshot. Runtime locks, manifests, outputs, and temporary files stay in writable storage. The intended upstream IDs/revisions remain in metadata even when local paths are used.

## Outputs

Each completed run contains original/resolved configuration, fingerprint, environment report, frozen manifest, exclusions, dataset audit, resumable JSONL checkpoints, `scores.csv`, failures, runtime, `metrics.json`, `metrics.md`, ROC curve, score histogram, score summary, and `run.log`. Peak allocated/reserved CUDA memory and per-sample runtime are in every score row.

## Interpretation and limitations

Poisoned Chalice separation can reflect much more than memorization: source/distribution shift (including The Stack Edu versus The Heap), collection time, licences, library/API usage, code length, duplicates or near-duplicates, and residual corpus differences. A short 32-word prefix emphasizes headers, comments, licences, and imports. True exposure is uncertain, and the released public test need not reproduce the original hidden leaderboard evaluation. These factors can make a classifier distinguish corpora rather than membership. Report results as a baseline diagnostic with complete provenance, not proof about an individual sample.

The 20+20 smoke run validates plumbing only. The 100+100 run has coarse low-FPR resolution and high statistical uncertainty. This repository intentionally omits SimMIA, Code-SimMIA, Code-Gap-K%, fine-tuning, other languages, 7B models, and full-dataset experiments.

## References

- [Gap-K% paper (ACL 2026)](https://aclanthology.org/2026.acl-long.1072/)
- [Official implementation, pinned commit](https://github.com/meaoww/gap-k/tree/46cd478fc7253131f5be8f8ef7897fbfdddef119)
