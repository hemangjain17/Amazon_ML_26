import os
import zipfile
import boto3
import urllib.request
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError


def ensure_dataset_extracted(local_dir: str) -> bool:
    """Extracts dataset.zip automatically if TSV dataset files are missing from local_dir."""
    if not local_dir:
        return False
    known_files = [
        "train/train_source1.tsv",
        "train/train_source2.tsv",
        "train/train_source3.tsv",
        "train/train_ground_truth.tsv",
        "test/test_source1.tsv",
        "test/test_source2.tsv",
        "test/test_source3.tsv",
    ]
    all_exist = all(
        os.path.exists(os.path.join(local_dir, f)) and os.path.getsize(os.path.join(local_dir, f)) > 100
        for f in known_files
    )
    if all_exist:
        return True

    search_dirs = [
        local_dir,
        os.path.dirname(local_dir),
        os.path.dirname(os.path.dirname(local_dir)),
    ]
    zip_path = None
    for d in search_dirs:
        candidate = os.path.join(d, "dataset.zip")
        if os.path.exists(candidate) and os.path.getsize(candidate) > 100:
            zip_path = candidate
            break

    if not zip_path:
        return False

    target_extract_dir = os.path.dirname(local_dir)
    print(f"📦 Extracting dataset archive from {zip_path} -> {target_extract_dir}...")
    os.makedirs(target_extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_extract_dir)
    print("✅ Dataset successfully extracted!")
    return True


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


def sync_directory_from_s3(bucket: str, s3_prefix: str, local_dir: str, region: str = "eu-north-1") -> bool:
    os.makedirs(local_dir, exist_ok=True)
    if ensure_dataset_extracted(local_dir):
        return True
    known_files = [
        "train/train_source1.tsv",
        "train/train_source2.tsv",
        "train/train_source3.tsv",
        "train/train_ground_truth.tsv",
        "test/test_source1.tsv",
        "test/test_source2.tsv",
        "test/test_source3.tsv",
    ]
    
    count = 0
    s3_client = boto3.client("s3")
    s3_unsigned = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name=region)

    for rel_file in known_files:
        target_path = os.path.join(local_dir, rel_file)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 100:
            count += 1
            continue

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        downloaded = False

        candidate_keys = [
            f"{s3_prefix}/{rel_file}".replace("//", "/"),
            f"dataset/{rel_file}".replace("//", "/"),
            f"{rel_file}".replace("//", "/")
        ]

        # Method A: Authenticated boto3
        for s3_key in candidate_keys:
            try:
                s3_client.download_file(bucket, s3_key, target_path)
                if os.path.exists(target_path) and os.path.getsize(target_path) > 100:
                    print(f"Downloaded via Authenticated S3: s3://{bucket}/{s3_key} -> {target_path}")
                    count += 1
                    downloaded = True
                    break
            except Exception:
                pass

        # Method B: Unsigned boto3 (Public S3 Bucket)
        if not downloaded:
            for s3_key in candidate_keys:
                try:
                    s3_unsigned.download_file(bucket, s3_key, target_path)
                    if os.path.exists(target_path) and os.path.getsize(target_path) > 100:
                        print(f"Downloaded via Unsigned Public S3: s3://{bucket}/{s3_key} -> {target_path}")
                        count += 1
                        downloaded = True
                        break
                except Exception:
                    pass

        # Method C: Direct HTTPS URL Download
        if not downloaded:
            for s3_key in candidate_keys:
                public_urls = [
                    f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}",
                    f"https://s3.{region}.amazonaws.com/{bucket}/{s3_key}",
                ]
                for url in public_urls:
                    try:
                        urllib.request.urlretrieve(url, target_path)
                        if os.path.exists(target_path) and os.path.getsize(target_path) > 100:
                            print(f"Downloaded via Public HTTPS: {url} -> {target_path}")
                            count += 1
                            downloaded = True
                            break
                    except Exception:
                        pass
                if downloaded:
                    break

        if not downloaded:
            print(f"Could not locate or download {rel_file} from S3 bucket {bucket}.")

    success = (count >= len(known_files))
    print(f"Synced {count}/{len(known_files)} dataset files from S3 -> {local_dir}")
    return success
