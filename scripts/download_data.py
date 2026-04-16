from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def download(url: str, target: Path, overwrite: bool = False) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return f"Skipped (exists): {target}"

    with urllib.request.urlopen(url) as response:
        data = response.read()
    target.write_bytes(data)
    size_mb = target.stat().st_size / (1024 * 1024)
    return f"Downloaded: {target} ({size_mb:.1f} MB)"


def extract_zip(zip_path: Path, output_dir: Path, overwrite: bool = False) -> str:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        return f"Skipped extract (exists): {output_dir}"

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)

    jpg_count = sum(1 for _ in output_dir.rglob("*.jpg"))
    return f"Extracted: {zip_path} -> {output_dir} ({jpg_count} jpg files)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download required MARS data files for MVP pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/mvp_config.json"))
    parser.add_argument("--overwrite", action="store_true", help="Re-download files even if present")
    parser.add_argument(
        "--include-raw-images-top",
        action="store_true",
        help="Download and extract raw top-view MARS image frames (large download)",
    )
    parser.add_argument(
        "--include-sample-video",
        action="store_true",
        help="Download the configured sample video from pose_inference_runtime.video_url",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]

    top_url = ds["top_keypoints_url"]
    front_url = ds["front_keypoints_url"]

    top_file = Path(ds["top_keypoints_file"])
    front_file = Path(ds["front_keypoints_file"])

    print(download(top_url, top_file, overwrite=args.overwrite))
    print(download(front_url, front_file, overwrite=args.overwrite))

    if args.include_raw_images_top:
        raw_top_url = ds.get("raw_images_top_url")
        raw_top_zip = ds.get("raw_images_top_zip")
        raw_top_dir = ds.get("raw_images_top_dir")
        if not raw_top_url or not raw_top_zip or not raw_top_dir:
            raise SystemExit("dataset.raw_images_top_url/raw_images_top_zip/raw_images_top_dir must be set in config")

        zip_path = Path(raw_top_zip)
        out_dir = Path(raw_top_dir)
        print(download(raw_top_url, zip_path, overwrite=args.overwrite))
        print(extract_zip(zip_path, out_dir, overwrite=args.overwrite))

    if args.include_sample_video:
        runtime_cfg = cfg.get("pose_inference_runtime", {})
        video_url = str(runtime_cfg.get("video_url", "")).strip()
        video_file = str(runtime_cfg.get("video_file", "")).strip()
        if not video_url or not video_file:
            raise SystemExit("pose_inference_runtime.video_url and pose_inference_runtime.video_file must be set")
        print(download(video_url, Path(video_file), overwrite=args.overwrite))


if __name__ == "__main__":
    main()
