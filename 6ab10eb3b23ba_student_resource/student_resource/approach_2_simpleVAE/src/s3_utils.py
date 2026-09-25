import os
import boto3
from botocore.exceptions import ClientError


def is_sagemaker_environment() -> bool:
    return os.path.exists("/opt/ml") or "SAGEMAKER_INTERNAL_IMAGE_URI" in os.environ


def upload_file_to_s3(local_file_path: str, bucket: str, s3_key: str) -> bool:
    if not os.path.exists(local_file_path):
        print(f"File not found: {local_file_path}")
        return False
    s3_client = boto3.client("s3")
    try:
        s3_client.upload_file(local_file_path, bucket, s3_key)
        print(f"Successfully uploaded {local_file_path} -> s3://{bucket}/{s3_key}")
        return True
    except ClientError as e:
        print(f"S3 Upload Error: {e}")
        return False


def download_file_from_s3(bucket: str, s3_key: str, local_file_path: str) -> bool:
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    s3_client = boto3.client("s3")
    try:
        s3_client.download_file(bucket, s3_key, local_file_path)
        print(f"Successfully downloaded s3://{bucket}/{s3_key} -> {local_file_path}")
        return True
    except ClientError as e:
        print(f"S3 Download Error: {e}")
        return False


def sync_directory_from_s3(bucket: str, s3_prefix: str, local_dir: str) -> None:
    os.makedirs(local_dir, exist_ok=True)
    s3_client = boto3.client("s3")
    count = 0
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    s3_key = obj["Key"]
                    if s3_key.endswith("/"):
                        continue
                    rel_path = os.path.relpath(s3_key, s3_prefix)
                    target_path = os.path.join(local_dir, rel_path)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    try:
                        s3_client.download_file(bucket, s3_key, target_path)
                        count += 1
                    except ClientError as e:
                        print(f"Failed to download {s3_key}: {e}")
    except Exception as e:
        print(f"ListObjectsV2 failed ({e}). Attempting direct file download fallback...")
        known_files = [
            "train/train_source1.tsv",
            "train/train_source2.tsv",
            "train/train_source3.tsv",
            "train/train_ground_truth.tsv",
            "test/test_source1.tsv",
            "test/test_source2.tsv",
            "test/test_source3.tsv",
        ]
        for k_file in known_files:
            s3_key = f"{s3_prefix}/{k_file}".replace("//", "/")
            target_path = os.path.join(local_dir, k_file)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            try:
                s3_client.download_file(bucket, s3_key, target_path)
                print(f"Downloaded s3://{bucket}/{s3_key} -> {target_path}")
                count += 1
            except Exception as dl_err:
                print(f"Could not download {s3_key}: {dl_err}")

    print(f"Synced {count} files from s3://{bucket}/{s3_prefix} -> {local_dir}")
