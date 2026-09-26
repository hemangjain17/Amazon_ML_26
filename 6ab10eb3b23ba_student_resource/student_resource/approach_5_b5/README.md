# Approach 5: B5-seg + LightGBM

This replaces the temporary Splink experiment. It is a country-aware blocking and two-stage LightGBM pipeline:

```text
B5-seg normalization
  -> inverted-index, sorted-neighbourhood and trigram blocking
  -> state compatibility and per-view top-k pruning
  -> Stage 1 LightGBM
  -> contextual rank/gap/support features
  -> Stage 2 LightGBM
  -> isotonic calibration
  -> expected F0.5 selection
  -> matching_results.tsv + candidate_pairs.tsv
```

The current checkout did not contain the original `approach.md` or EXP-FINAL implementation, so this package is a compatible implementation of the described contract rather than a literal copy of unavailable code. To reproduce the exact historical 69/80 feature model, copy the original B5 feature and decision modules into this directory and use them in `run_pipeline.py`.

## Kaggle

```python
%pip install -q -r /kaggle/working/Amazon_ML_26/6ab10eb3b23ba_student_resource/student_resource/approach_5_b5/requirements.txt
!python /kaggle/working/Amazon_ML_26/6ab10eb3b23ba_student_resource/student_resource/approach_5_b5/run_pipeline.py --dry-run --s1-limit 10000 --pool-limit 100000
!python /kaggle/working/Amazon_ML_26/6ab10eb3b23ba_student_resource/student_resource/approach_5_b5/run_pipeline.py --output-dir /kaggle/working/output --s1-chunk-size 10000 --max-train-pairs 2000000 --threads 4
```

To deliver the completed files through Telegram, set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in the Kaggle notebook environment before running. The pipeline sends line-preserving chunks in this order: all `matching_results.tsv` chunks first, then all `candidate_pairs.tsv` chunks. Use `--no-telegram` to disable uploads or `--telegram-chunk-mb 45` to change the chunk size.

The runner streams test inference in S1 chunks and releases the training tables before test scoring. It does not create embedding caches or intermediate pair files; only the two required TSV outputs are written. `--max-train-pairs` bounds the in-memory training matrix, while `--s1-chunk-size` bounds test pair and feature memory.

Progress bars cover normalization, token counting, index construction, candidate generation, context-feature construction, and output writing. A smoke run is typically 5-20 minutes. The full run is approximately 3-8 hours, depending mainly on candidate volume and CPU contention. Budget 30 GB RAM and at least 25 GB free disk.

This implementation is CPU-bound. The standard Kaggle LightGBM wheel does not provide a reliable two-GPU execution path, and string normalization/blocking remains CPU work even with GPU LightGBM. `--threads 4` is the safe starting point for Kaggle; increase it only after checking RAM. Using both GPUs would require replacing the string and LightGBM stages with a CUDA-supported implementation and would be a separate architecture change.

Required files are the standard six TSVs under `dataset/train/` and `dataset/test/`. The full run is CPU/RAM bound during blocking and LightGBM; GPU is not required. Keep at least 25 GB free disk and 30 GB RAM for the stated 10.3M-row pool.
