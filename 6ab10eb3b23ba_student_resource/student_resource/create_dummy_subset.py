import os
import pandas as pd

def create_dummy_subset(
    src_dir: str,
    target_dir: str,
    num_train_s1: int = 3000,
    num_test_s1: int = 1000,
):
    """Creates a self-contained dummy subset dataset for quick local execution and scoring."""
    os.makedirs(os.path.join(target_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "test"), exist_ok=True)

    print(f"Reading S1 records from {src_dir}...")
    train_s1_full = pd.read_csv(os.path.join(src_dir, "train", "train_source1.tsv"), sep="\t")
    gt_full = pd.read_csv(os.path.join(src_dir, "train", "train_ground_truth.tsv"), sep="\t")

    # Sample train S1 and test S1 (disjoint)
    train_s1_sub = train_s1_full.iloc[:num_train_s1].copy()
    test_s1_sub = train_s1_full.iloc[num_train_s1 : num_train_s1 + num_test_s1].copy()

    train_s1_ids = set(train_s1_sub["entity_id"])
    test_s1_ids = set(test_s1_sub["entity_id"])

    # Ground truth subset
    gt_train_sub = gt_full[gt_full["source1_entity_id"].isin(train_s1_ids)].copy()
    gt_test_sub = gt_full[gt_full["source1_entity_id"].isin(test_s1_ids)].copy()

    # Collect all referenced S2 and S3 entity IDs for train and test
    def extract_matched_ids(gt_df):
        s2_ids, s3_ids = set(), set()
        for matches_str in gt_df["matched_entity_ids"].dropna():
            for m_id in str(matches_str).split(","):
                m_id = m_id.strip()
                if m_id.startswith("S2-"):
                    s2_ids.add(m_id)
                elif m_id.startswith("S3-"):
                    s3_ids.add(m_id)
        return s2_ids, s3_ids

    train_s2_match_ids, train_s3_match_ids = extract_matched_ids(gt_train_sub)
    test_s2_match_ids, test_s3_match_ids = extract_matched_ids(gt_test_sub)

    print(f"Matched IDs in Train: S2={len(train_s2_match_ids)}, S3={len(train_s3_match_ids)}")
    print(f"Matched IDs in Test: S2={len(test_s2_match_ids)}, S3={len(test_s3_match_ids)}")

    # Load S2 and S3 files (using chunking or full read if fast enough)
    print("Loading S2 and S3 full sources to filter matched subset + distractors...")
    s2_full = pd.read_csv(os.path.join(src_dir, "train", "train_source2.tsv"), sep="\t", nrows=100000)
    s3_full = pd.read_csv(os.path.join(src_dir, "train", "train_source3.tsv"), sep="\t", nrows=100000)

    train_s2_sub = s2_full[s2_full["entity_id"].isin(train_s2_match_ids) | (s2_full.index < 10000)].copy()
    train_s3_sub = s3_full[s3_full["entity_id"].isin(train_s3_match_ids) | (s3_full.index < 10000)].copy()

    test_s2_sub = s2_full[s2_full["entity_id"].isin(test_s2_match_ids) | (s2_full.index.isin(range(10000, 20000)))].copy()
    test_s3_sub = s3_full[s3_full["entity_id"].isin(test_s3_match_ids) | (s3_full.index.isin(range(10000, 20000)))].copy()

    # Save to target_dir/train
    train_s1_sub.to_csv(os.path.join(target_dir, "train", "train_source1.tsv"), sep="\t", index=False)
    train_s2_sub.to_csv(os.path.join(target_dir, "train", "train_source2.tsv"), sep="\t", index=False)
    train_s3_sub.to_csv(os.path.join(target_dir, "train", "train_source3.tsv"), sep="\t", index=False)
    gt_train_sub.to_csv(os.path.join(target_dir, "train", "train_ground_truth.tsv"), sep="\t", index=False)

    # Save to target_dir/test
    test_s1_sub.to_csv(os.path.join(target_dir, "test", "test_source1.tsv"), sep="\t", index=False)
    test_s2_sub.to_csv(os.path.join(target_dir, "test", "test_source2.tsv"), sep="\t", index=False)
    test_s3_sub.to_csv(os.path.join(target_dir, "test", "test_source3.tsv"), sep="\t", index=False)
    gt_test_sub.to_csv(os.path.join(target_dir, "test", "test_ground_truth.tsv"), sep="\t", index=False)

    print(f"=== DUMMY SUBSET CREATED SUCCESSFULLY IN {target_dir} ===")
    print(f"Train: S1={len(train_s1_sub)}, S2={len(train_s2_sub)}, S3={len(train_s3_sub)}, GT={len(gt_train_sub)}")
    print(f"Test:  S1={len(test_s1_sub)}, S2={len(test_s2_sub)}, S3={len(test_s3_sub)}, GT={len(gt_test_sub)}")

if __name__ == "__main__":
    src = r"c:\Users\heman\OneDrive\Desktop\Hemang\Amazon_ML_26\6ab10eb3b23ba_student_resource\student_resource\dataset"
    target = r"c:\Users\heman\OneDrive\Desktop\Hemang\Amazon_ML_26\6ab10eb3b23ba_student_resource\student_resource\dummy_folder\dataset"
    create_dummy_subset(src, target, num_train_s1=2000, num_test_s1=500)
