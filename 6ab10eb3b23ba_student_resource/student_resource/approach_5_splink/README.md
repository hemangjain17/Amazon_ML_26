# Approach 5: Splink + DuckDB Entity Resolution

This approach is a CPU-friendly probabilistic baseline for the challenge. It keeps the existing country-aware data contract, creates capped blocking keys, uses Splink/DuckDB for candidate generation and comparison scoring, selects the threshold on a held-out labelled split, and writes the official submission files.

## Kaggle setup

From a notebook cell:

```python
%pip install -q -r /kaggle/working/student_resource/approach_5_splink/requirements.txt
!python /kaggle/working/student_resource/approach_5_splink/run_pipeline.py --dry-run --subset-size 1000
```

Adjust the path if the repository is mounted under a different Kaggle input directory. For a full run:

```python
!python /kaggle/working/student_resource/approach_5_splink/run_pipeline.py
```

Outputs are written to `/kaggle/working/output/`:

- `candidate_pairs.tsv`: every pair emitted by the Splink blocking rules;
- `matching_results.tsv`: thresholded matches after the pool-record uniqueness policy.

## Method

The blocker uses country plus capped rare keys for name tokens, address tokens, and house numbers. Frequent or missing keys are represented as SQL `NULL` and never create an all-missing Cartesian product. Splink estimates non-match comparison probabilities from random sampling and match probabilities from the labelled training split.

The training data is split into fitting and validation partitions. Splink parameters are fitted on the first partition; the probability threshold is selected on the second. The final test run uses parameters retrained on all labelled training records while keeping the validation threshold frozen.

The official task is one-to-many from Source 1: one S1 row may have multiple matches, while a Source 2/3 row is assigned to at most one S1 row. The assignment code enforces the latter globally by retaining the highest-probability owner for each pool entity.

## Required data

The Kaggle dataset must contain:

```text
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

The data columns must include `entity_id`, `business_name`, `business_address`, and `country`; the ground truth must include `source1_entity_id` and `matched_entity_ids`.

## Tuning knobs

```text
--token-cap       maximum combined frequency for a name/address token (default 300)
--house-cap       maximum pool frequency for a house number (default 1200)
--max-comparisons abort before Splink if the estimated blocking work is too large
--validation-fraction labelled S1 fraction reserved for threshold selection (default 0.2)
--subset-size     deterministic S1 subset for a cheap smoke run
```

Run the dry run before the full job. Compare validation blocking recall, candidate count, macro F0.5, and wall time against the current EXP-FINAL blocker before replacing it in production.
