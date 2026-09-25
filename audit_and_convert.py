import os
import json
import time
import pandas as pd
import numpy as np

BASE_DIR = r"c:\Users\heman\OneDrive\Desktop\Hemang\Amazon_ML_26\6ab10eb3b23ba_student_resource\student_resource"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CSV_DIR = os.path.join(BASE_DIR, "dataset_csv")

os.makedirs(os.path.join(CSV_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(CSV_DIR, "test"), exist_ok=True)

files_to_process = [
    ("train", "train_source1.tsv", "train_source1.csv"),
    ("train", "train_source2.tsv", "train_source2.csv"),
    ("train", "train_source3.tsv", "train_source3.csv"),
    ("train", "train_ground_truth.tsv", "train_ground_truth.csv"),
    ("test", "test_source1.tsv", "test_source1.csv"),
    ("test", "test_source2.tsv", "test_source2.csv"),
    ("test", "test_source3.tsv", "test_source3.csv"),
]

stats = {}

print("=== Starting TSV to CSV Conversion and Data Audit ===")
for sub, tsv_name, csv_name in files_to_process:
    tsv_path = os.path.join(DATASET_DIR, sub, tsv_name)
    csv_path = os.path.join(CSV_DIR, sub, csv_name)
    
    if not os.path.exists(tsv_path):
        print(f"Skipping {tsv_path} (not found)")
        continue

    file_size_mb = os.path.getsize(tsv_path) / (1024 * 1024)
    t0 = time.time()
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    
    # Save CSV
    df.to_csv(csv_path, index=False, encoding="utf-8")
    t1 = time.time()
    
    file_stats = {
        "file_size_mb": round(file_size_mb, 2),
        "rows": len(df),
        "columns": list(df.columns),
        "null_or_empty_counts": {col: int((df[col].str.strip() == "").sum()) for col in df.columns},
        "time_taken_sec": round(t1 - t0, 2)
    }
    
    if "country" in df.columns:
        file_stats["country_distribution"] = df["country"].value_counts().to_dict()
        
    stats[f"{sub}/{tsv_name}"] = file_stats
    print(f"Converted {sub}/{tsv_name} -> {csv_name} ({file_stats['rows']} rows, {file_stats['file_size_mb']} MB) in {file_stats['time_taken_sec']}s")

# Analyze ground truth
gt_path = os.path.join(CSV_DIR, "train", "train_ground_truth.csv")
if os.path.exists(gt_path):
    gt_df = pd.read_csv(gt_path, dtype=str, keep_default_na=False)
    gt_df["matched_list"] = gt_df["matched_entity_ids"].apply(lambda x: [i.strip() for i in x.split(",") if i.strip()])
    gt_df["match_count"] = gt_df["matched_list"].apply(len)
    
    s2_matches = gt_df["matched_list"].apply(lambda l: sum(1 for item in l if item.startswith("S2-")))
    s3_matches = gt_df["matched_list"].apply(lambda l: sum(1 for item in l if item.startswith("S3-")))
    
    gt_stats = {
        "total_s1_entities": len(gt_df),
        "singletons_count": int((gt_df["match_count"] == 0).sum()),
        "singletons_pct": round(float((gt_df["match_count"] == 0).mean() * 100), 2),
        "entities_with_matches": int((gt_df["match_count"] > 0).sum()),
        "total_matches_ground_truth": int(gt_df["match_count"].sum()),
        "avg_matches_per_entity": round(float(gt_df["match_count"].mean()), 3),
        "s2_matches_count": int(s2_matches.sum()),
        "s3_matches_count": int(s3_matches.sum()),
        "match_count_distribution": gt_df["match_count"].value_counts().head(10).to_dict()
    }
    stats["ground_truth_analysis"] = gt_stats
    print("\n--- Ground Truth Summary ---")
    print(json.dumps(gt_stats, indent=2))

with open(os.path.join(CSV_DIR, "audit_summary.json"), "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nAudit completed. CSV files saved in: {CSV_DIR}")
