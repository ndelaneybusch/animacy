import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download expected activation summaries from S3."
    )
    parser.add_argument(
        "--responses_dir",
        type=str,
        required=True,
        help="Directory containing the source JSON response files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Local directory to save downloaded files.",
    )
    parser.add_argument(
        "--s3_bucket", type=str, default="r7y28kemyg", help="S3 bucket name."
    )
    parser.add_argument(
        "--s3_prefix",
        type=str,
        required=True,
        help="Prefix in the bucket (e.g., animacy/results/activations/data/gemma-3-27b-it/without_sys/).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="List of layer indices to download.",
    )
    parser.add_argument(
        "--s3_region", type=str, default="us-ca-2", help="S3 region name."
    )
    parser.add_argument(
        "--endpoint_url",
        type=str,
        default="https://s3api-us-ca-2.runpod.io",
        help="Endpoint URL for S3.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Number of parallel download threads.",
    )
    return parser.parse_args()


def get_expected_files(responses_dir: Path, layers: list[int]) -> set[str]:
    """
    Scan response JSONs and generate expected filenames.
    Returns a set of filenames (e.g., 'role_task_0_layer10.json').
    """
    expected_files = set()
    json_files = list(responses_dir.glob("*.json"))

    print(f"Scanning {len(json_files)} response files for expected activations...")

    for json_file in tqdm(json_files, desc="Parsing responses"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                role_name = item.get("role_name")
                task_name = item.get("task_name")
                sample_idx = item.get("sample_idx")

                if role_name is None or task_name is None or sample_idx is None:
                    continue

                for layer in layers:
                    filename = f"{role_name}_{task_name}_{sample_idx}_layer{layer}.json"
                    expected_files.add(filename)

        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    return expected_files


def download_file(
    s3_client, bucket: str, key: str, local_path: Path
) -> tuple[str, str]:
    """
    Download a single file from S3.
    Returns (filename, status).
    """
    try:
        s3_client.download_file(bucket, key, str(local_path))
        return local_path.name, "downloaded"
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            return local_path.name, "missing"
        else:
            return local_path.name, f"error: {error_code}"
    except Exception as e:
        return local_path.name, f"error: {str(e)}"


def main():
    args = parse_args()

    responses_dir = Path(args.responses_dir)
    output_dir = Path(args.output_dir)

    if not responses_dir.exists():
        print(f"Error: Responses directory {responses_dir} does not exist.")
        sys.exit(1)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Identify expected files
    expected_filenames = get_expected_files(responses_dir, args.layers)
    print(f"Found {len(expected_filenames)} expected activation files.")

    # 2. Filter out files that already exist locally
    files_to_download = []
    for filename in expected_filenames:
        local_path = output_dir / filename
        if not local_path.exists():
            files_to_download.append(filename)

    print(f"Files already present: {len(expected_filenames) - len(files_to_download)}")
    print(f"Files to download: {len(files_to_download)}")

    if not files_to_download:
        print("All files already exist locally.")
        sys.exit(0)

    # 3. Initialize S3 client
    s3 = boto3.client(
        "s3",
        endpoint_url=args.endpoint_url,
        region_name=args.s3_region,
        # Credentials should be in env vars or ~/.aws/credentials
    )

    # 4. Download in parallel
    print(f"Starting download with {args.max_workers} threads...")

    # Ensure prefix ends with /
    prefix = args.s3_prefix
    if not prefix.endswith("/"):
        prefix += "/"

    # Check if we need to append 'summaries/'
    # If the user didn't include it, and we know the standard structure has it.
    if "summaries" not in prefix:
        print(
            "Note: 'summaries' not found in prefix. Appending 'summaries/' to match standard structure."
        )
        prefix += "summaries/"

    print(f"Using S3 Prefix: {prefix}")

    success_count = 0
    missing_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        first_key = True
        for filename in files_to_download:
            key = f"{prefix}{filename}"
            if first_key:
                print(f"Debug: First download attempt key: {key}")
                first_key = False

            local_path = output_dir / filename
            future = executor.submit(download_file, s3, args.s3_bucket, key, local_path)
            futures[future] = filename

        for future in tqdm(
            as_completed(futures), total=len(files_to_download), desc="Downloading"
        ):
            filename, status = future.result()

            if status == "downloaded":
                success_count += 1
            elif status == "missing":
                missing_count += 1
            else:
                error_count += 1
                if error_count <= 10:  # Print first 10 errors
                    print(f"Failed to download {filename}: {status}")

    print("\nDownload Summary:")
    print(f"Successfully downloaded: {success_count}")
    print(f"Missing on S3: {missing_count}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    main()
