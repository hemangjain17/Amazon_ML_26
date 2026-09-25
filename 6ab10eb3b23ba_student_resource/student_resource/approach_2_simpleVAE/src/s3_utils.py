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


def sync_directory_to_s3(local_dir: str, bucket: str, s3_prefix: str) -> None:
    s3_client = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            try:
                s3_client.upload_file(local_path, bucket, s3_key)
            except ClientError as e:
                print(f"Failed to upload {local_path}: {e}")
    print(f"Synced directory {local_dir} -> s3://{bucket}/{s3_prefix}")
