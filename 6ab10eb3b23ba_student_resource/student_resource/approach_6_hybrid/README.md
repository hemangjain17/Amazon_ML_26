# Approach 6: B5 + dual-GPU E5 + LightGBM

Approach 6 combines the fast B5 blocker with `intfloat/multilingual-e5-small` pair similarity. B5 remains CPU-side; E5 encoding is split across the requested GPUs; LightGBM consumes the B5 features plus E5 cosine similarity.

```text
B5 normalization and blocking
  -> bounded candidate pairs
  -> E5-small query/passage embeddings on GPU 0 + GPU 1
  -> B5 string features + E5 cosine
  -> LightGBM Stage 1
  -> rank/gap/context features
  -> LightGBM Stage 2
  -> isotonic calibration and global pool ownership
  -> matching_results.tsv
  -> candidate_pairs.tsv
  -> Telegram: matching first, candidates second
```

## Kaggle commands

```python
%pip install -q -r /kaggle/working/Amazon_ML_26/6ab10eb3b23ba_student_resource/student_resource/approach_6_hybrid/requirements.txt
```

The first run needs Kaggle Internet access to download the E5 model. Store `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` as Kaggle Secrets if artifact delivery is wanted.

Smoke run:

```python
!python /kaggle/working/Amazon_ML_26/6ab10eb3b23ba_student_resource/student_resource/approach_6_hybrid/run_pipeline.py --dry-run --s1-limit 5000 --pool-limit 100000 --s1-chunk-size 1000 --max-train-pairs 100000 --devices 0,1 --output-dir /kaggle/working/output6_smoke --no-telegram
```

Full run:

```python
!python /kaggle/working/Amazon_ML_26/6ab10eb3b23ba_student_resource/student_resource/approach_6_hybrid/run_pipeline.py --s1-chunk-size 5000 --max-train-pairs 2000000 --embedding-batch-size 256 --threads 4 --devices 0,1 --telegram-chunk-mb 45 --output-dir /kaggle/working/output6
```

`--devices 0,1` uses both GPUs when two CUDA devices are visible. With one GPU, the encoder falls back to one device; with no GPU, it uses CPU. The B5 blocker and LightGBM remain CPU work.

## Resources

The model download is roughly 1 GB. E5 inference adds substantial time: expect 5-15 minutes for the smoke run and roughly 5-12 hours for the full run, depending on candidate volume and GPU/CPU throughput. Use 30 GB RAM and at least 25 GB free disk. No embedding cache is written; embeddings are discarded after each feature batch/chunk.
