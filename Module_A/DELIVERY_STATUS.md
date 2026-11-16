# Module A — Y-side delivery status (2026-11-16)

## Implemented and verified

- Single entry point `main.py` orchestrates generation, continuous logging, resume, and automated partial/full audit.
- Source generator is unmodified from the user's tested Kaggle version, SHA256 `3595b427adf520d56b8cf33a396174156c4bf04408d0cdae8350105e8e0538a2`.
- Full train2017 GT-box run: **4,682 / 4,682 shards**, **149,813 / 149,813 unique people**, **2,546,821** true Gain values; **0 Failed**; independent validation report **PASS**, issues **0**.
- Controlled simulated shard failure → status + JSONL error logging → continue to next shard: passed. Independent earlier retry from failed to done: passed. Interrupted full generation resumed at 173 done shards and completed.
- Saved label schema, ID alignment, OKS-before, Gain arithmetic and NPZ integrity checks passed; independent audit does not re-infer every Pj.
- Optional `review_a_labels.py` sample model re-inference and alpha=0 checks passed on sampled batches: 6 sampled gain candidates; no claim of full second pass.
- **New:** `summarize_a_labels.py` was tested against the exact archived 4,682 NPZ shards and reproduced all expected sign counts and mean gain; it does not use the GPU or modify data.
- Full P0 diagnostic and 6 positive/negative candidate examples, plus curated charts and CSVs, are under `results/module_a`.

## Pending / not yet accepted

1. Confirm with NIE/TANG the source of cached P0 vs TANG-interface reconstructed P0 discrepancies. The source is the **same NIE cached features**, not a separately saved TANG P0. The original NIE SimCC logits were not retained, so the argmax-tie hypothesis is **not conclusively proven**. Large differences (maximum coordinate ~295.96px and maximum per-instance OKS ~0.12142) merit explicit review.
2. Agree whether existing official `oks_before = OKS(NIE cached P0, GT)` is the accepted baseline given the possible cross-path precision difference.
3. Commit the reviewed generator/checker/shared OKS version to the existing Git repository. Confirm owner approval for any L-owned shared code changes.
4. Execute full-safe finalization (not done in the archived results). Current archived `metadata.json`: `complete=false`, `count=0`, `shards=[]` even though full NPZ coverage is present. Only finalizer may write real commit, SHA256 manifest and `complete=true` after the audit passes.
5. Publish/version the **updated** full label dataset separately from this code overlay; provide dataset URI and checksum to N. N then performs its own ID/input alignment acceptance before training.

## Artifacts / scope

- Source + curated results: this ZIP / team Git branch.
- Full data (not in this ZIP): `Module_A_Train2017_Labels.zip` (current **pre-finalization backup**).
- Detailed logs (not in this ZIP): `Module_A_Analysis_and_Logs.zip`.
- NO trained Gain Predictor model or Val AP/AR results are claimed; these are N's later tasks.
