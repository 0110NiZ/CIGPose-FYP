# Module A — True OKS Gain Labels (Y-side)

This is a **repository-relative source and curated-evidence overlay** for the team's existing CIGPose-FYP-ModuleA repository. It is **not a standalone copy** of CIGPose/MMPose, a replacement for N's training implementation, or a complete Train2017 label dataset.

## Ownership and interfaces

- **Y:** true candidates, true Gain label generator, failure logging/resume, independent label audit, supplementary label review.
- **N:** A input encoding, Gain Predictor, training, detection-box Val2017 inference and model evaluation; joins labels to shared-cache input **by `(dataset_split, annotation_id)` and checks `image_id`**, never by row position.
- **Shared dependencies:** TANG's feature intervention and downstream prediction/decoding interface; L's `oks_utility.py`. L owns that shared OKS file: reconcile versions and permissions before modifying it in the team repository.

N's original handoff in `module_a_handoff_zh.md` and `module_a.md` records the original interface contract; its historical "no labels yet" comments predate the successful Y-side full generation.

## Main workflow: generate and audit

Run from the existing repository's CIGPose Python environment:

```bash
python -u main.py \
  --repo /absolute/path/to/CIGPose-FYP-ModuleA \
  --cache /absolute/path/to/shared_cache/train2017 \
  --checkpoint /absolute/path/to/cigpose-m_coco_256x192.pth \
  --output /absolute/path/to/train_a_labels \
  --device cuda:0 \
  --max-shards 0 \
  --no-finalize
```

- `--max-shards 10` for a **separate** smoke-test directory; `0` for the full train2017 cache. The last tested full run had 4,682 source shards and 149,813 people.
- `--no-finalize` runs the full independent audit but does **not** set `metadata.complete=true` or require a Git commit. It was used for the archived full run.
- For interrupted runs, repeat exactly the same command with the **same output directory and unchanged generator and shared OKS code hashes**. The generator verifies and skips recorded successful shards. Do not run concurrent writers to the same output.
- The CLI arguments override `module_a_run_config.json` if present. Use `module_a_run_config.example.json` as a template; do not commit local paths.
- `main.py` orchestrates the subprocesses. The verified P0/Pj/OKS/Gain computations remain inside `build_a_labels.py`, not in duplicate copies inside the entry point.
- A temporary `AssertionError` may currently be classified as recoverable at shard level. Review repeated failures for systemic causes before relying on the continue-on-error behavior.

## Mathematical/data contract

Each person gets **17 independent candidates**, one for each COCO joint 0–16, with `alpha=1`. Every candidate starts from the **same untouched cached features**. The candidate positions are returned to original-image coordinates and scored against the same GT keypoints, visibility and area.

```text
oks_before = OKS(NIE_cached_P0, matched_GT)
oks_after[j] = OKS(Pj_from_independent_alpha1_intervention, matched_GT)
gain[j] = oks_after[j] - oks_before
```

**The official P0 is NIE's cached `P_0_coords`.** Reconstructed P0 is an additional check: we decode **unmodified features from the NIE cache using TANG's public downstream prediction/decoding interfaces**. This is not a comparison of two separately saved NIE/TANG P0 datasets. The reconstructed and cached P0 can differ, including some large outliers, and the reason has **not** been conclusively established. Never claim all differences are adjacent SimCC bins; consult `results/module_a/review/REVIEW_REPORT.md`.

Every formal output shard `shard-*.npz` contains **only** `image_id` int64 `[N]`, `annotation_id` int64 `[N]`, `oks_before` float `[N]`, `oks_after` float `[N,17]`, and signed `gain` float `[N,17]`. Missing or failed cases cannot be silently removed or filled with zeros. Negative Gain values are retained. Ground truth and true gains are **not** predictor inputs during inference.

## Read-only postprocessing / reproducing the results

Recompute full statistics and figures from the saved NPZ shards (no GPU required):

```bash
python mmpose/analysis/fyp/summarize_a_labels.py \
  --output /absolute/path/to/train_a_labels --expect-full
```

Outputs: `gain_statistics/gain_summary.json`, `per_joint_gain.csv`, `representative_case_ids.csv`, `GAIN_RESULTS_SUMMARY.md`, and three PNG charts. This script has been run against all 4,682 archived label shards and reproduced the recorded label counts and Gain mean.

Optional deeper model re-inference and P0 review (GPU/model needed):

```bash
python mmpose/analysis/fyp/review_a_labels.py \
  --repo /absolute/path/to/CIGPose-FYP-ModuleA \
  --cache /absolute/path/to/shared_cache/train2017 \
  --checkpoint /absolute/path/to/cigpose-m_coco_256x192.pth \
  --output /absolute/path/to/train_a_labels \
  --device cuda:0
```

For actual photographic pose overlays, supply `--images-dir /path/to/COCO/train2017`; otherwise diagrams are **skeleton-only original-coordinate plots**. `review_a_labels.py` is a separate *optional* postprocessing command, not currently invoked by `main.py`. See `README_review_a_labels.md`.

## Files, logs, validation, handoff

| Path | Role |
| --- | --- |
| `main.py` | One-command generator + audit orchestrator with sequential progress windows and timings |
| `mmpose/analysis/fyp/build_a_labels.py` | Frozen-CIGPose true Gain generation, P0 diagnostic, resume, failed statuses and JSONL history |
| `mmpose/analysis/fyp/run_a_labels_logged.py` | Generation invocation and full log capture |
| `mmpose/analysis/fyp/check_a_labels.py` | Independent structural / ID / OKS-before / gain / full-coverage audit |
| `mmpose/analysis/fyp/finalize_a_labels.py` | Full audit, Git revision / hashes, `metadata.complete=true` only on verified full delivery |
| `mmpose/analysis/fyp/review_a_labels.py` | Optional model-based sample re-inference and P0 visual review |
| `mmpose/analysis/fyp/summarize_a_labels.py` | Reproducible post-hoc Gain statistics (read-only) |
| `mmpose/analysis/fyp/oks_utility.py` | **Shared L-owned implementation**: check against current team code before merging |
| `results/module_a/` | Curated evidence only; no large NPZ training dataset or raw logs |

Model generation requires the team's installed PyTorch/MMPose/CIGPose environment, the frozen checkpoint and the exact shared cache. The standalone statistics tool needs NumPy and (for charts) Matplotlib; the main progress UI uses `tqdm` when present.

Full run logs, `p0_diagnostics/`, `failure_history.jsonl` (if failures occurred), `status.json`, NPZ labels, checkpoint and raw shared cache are **not** code to commit. Share the verified label directory separately as a versioned dataset after acceptance.

### Current evidence (2026-11-16)

- 4,682/4,682 shards, 149,813/149,813 persons, **2,546,821** signed candidate gains; zero failed shards; independent audit reports zero issues.
- Saved Gain signs: **828,848 positive; 1,694,009 negative; 23,964 zero**. Mean true Gain -0.01014629. Oracle-best statistics are *offline label analysis*, not a trained A model result.
- Independent full audit checks manifest, IDs, count, NPZ schema, file hashes, cached P0 OKS and arithmetic. It does **not** rerun every model candidate. The optional review successfully re-inferred 6 sampled candidates.
- P0 provenance mismatch is **OPEN**: some original cached P0 coordinates differ from TANG-interface reconstruction; max recorded coordinate difference ~295.96 px and max instance OKS difference ~0.12142. Need NIE/TANG confirmation. Do not claim that all differences come from adjacent SimCC argmax bins.
- **Archived labels were generated with `--no-finalize`.** Although full coverage and audit passed, the archived `metadata.json` still has `complete=false`, `count=0`, and an empty shard manifest. It is not yet a formally accepted training release.

## Final acceptance / release

Before calling the label dataset final: review/record the P0 issue and acceptance decision, commit the generator/checker and shared OKS version in the existing team repo, run `finalize_a_labels.py` against the full output, and verify the final metadata manifest. Finalization uses the real Git commit and hashes and reruns full independent audit. Once successful, repack the **updated** metadata with all 4,682 label NPZ shards as a separate versioned dataset for N. Do not force `complete=true` manually.

See `docs/fyp/DELIVERY_STATUS.md` for current checklist, and `docs/fyp/GITHUB_INTEGRATION.md` before merging into the team's branch.
