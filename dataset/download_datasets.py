"""
Dataset Downloader for the AI Cattle Analysis System (Phase 2).

Downloads every dataset source declared under ``datasets.sources`` in
``configs/model_config.yaml`` into ``data/raw/<source-name>/``.

Supported source types:
  - kaggle       : Kaggle datasets via the ``kaggle`` CLI (requires API credentials)
  - roboflow     : Roboflow Universe exports via the Roboflow REST API
                   (requires the ``ROBOFLOW_API_KEY`` environment variable)
  - coco         : Category-filtered COCO 2017 subsets (annotations + images)
  - http         : Direct archive/file downloads over HTTP(S)
  - github       : Git repository clones
  - huggingface  : Hugging Face Hub datasets via the ``datasets`` library

Every run writes ``data/raw/download_manifest.json`` describing the outcome of
each source so the unification step (``dataset/unify_datasets.py``) can verify
provenance.

Usage:
    python -m dataset.download_datasets                     # all sources
    python -m dataset.download_datasets --source cattle_breeds_kaggle
    python -m dataset.download_datasets --list
    python -m dataset.download_datasets --max-images 500    # cap COCO pulls
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "model_config.yaml"

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_SPLITS = ("train2017", "val2017")

DOWNLOAD_CHUNK_SIZE = 1 << 20  # 1 MiB
HTTP_TIMEOUT = 60
MAX_RETRIES = 4
COCO_IMAGE_WORKERS = 8

# Roboflow export-format identifiers keyed by the format names used in the config.
ROBOFLOW_FORMATS = {"yolo": "yolov8", "coco": "coco", "voc": "voc"}

logger = logging.getLogger("dataset.download")


# ─────────────────────────────── helpers ────────────────────────────────


def setup_logging(logs_dir: Path) -> None:
    """Configure console + file logging (idempotent)."""
    if logger.handlers:
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(logs_dir / "download_datasets.log")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load the master YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def count_files(directory: Path) -> int:
    """Recursively count regular files under a directory."""
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if p.is_file())


def download_file(url: str, dest: Path, retries: int = MAX_RETRIES) -> Path:
    """
    Stream a URL to ``dest`` with retries and exponential backoff.

    Downloads to a ``.part`` temp file and renames atomically on success so a
    partially written file is never mistaken for a complete download.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"Already downloaded, skipping: {dest.name}")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=HTTP_TIMEOUT) as resp:
                resp.raise_for_status()
                expected = int(resp.headers.get("Content-Length", 0))
                written = 0
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        f.write(chunk)
                        written += len(chunk)
                if expected and written != expected:
                    raise IOError(f"Incomplete download: {written}/{expected} bytes")
            tmp.rename(dest)
            return dest
        except Exception as e:  # noqa: BLE001 — retry any transport failure
            last_error = e
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                wait = 2 ** attempt
                logger.warning(f"Download failed ({e}); retry {attempt}/{retries - 1} in {wait}s")
                time.sleep(wait)
    raise IOError(f"Failed to download {url}: {last_error}")


def extract_archive(archive: Path, dest: Path) -> None:
    """Extract a .zip / .tar(.gz|.bz2|.xz) archive into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")
    else:
        raise ValueError(f"Unsupported archive format: {archive.name}")


def run_command(cmd: List[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    """Run a subprocess, raising on non-zero exit with captured stderr."""
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)


def _result(source: Dict[str, Any], status: str, path: Path,
            error: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    """Build a uniform per-source manifest entry."""
    entry: Dict[str, Any] = {
        "name": source.get("name", "unnamed"),
        "type": source.get("type", "unknown"),
        "status": status,
        "path": str(path),
        "file_count": count_files(path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        entry["error"] = error
    entry.update(extra)
    return entry


# ───────────────────────────── source handlers ──────────────────────────────


def download_kaggle(source: Dict[str, Any], dest: Path) -> Dict[str, Any]:
    """Download a Kaggle dataset with the ``kaggle`` CLI (``--unzip``)."""
    dataset_id = source.get("dataset_id")
    if not dataset_id:
        return _result(source, "failed", dest, error="Missing 'dataset_id' in source config")

    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists() and not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return _result(
            source, "failed", dest,
            error="Kaggle credentials not found. Place kaggle.json in ~/.kaggle/ "
                  "or set KAGGLE_USERNAME and KAGGLE_KEY.",
        )

    dest.mkdir(parents=True, exist_ok=True)
    try:
        run_command(["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(dest), "--unzip"])
    except FileNotFoundError:
        return _result(source, "failed", dest, error="'kaggle' CLI not installed (pip install kaggle)")
    except subprocess.CalledProcessError as e:
        return _result(source, "failed", dest, error=(e.stderr or str(e)).strip())
    except subprocess.TimeoutExpired:
        return _result(source, "failed", dest, error="Kaggle download timed out")
    return _result(source, "downloaded", dest, dataset_id=dataset_id)


def _parse_roboflow_ref(source: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Resolve workspace/project/version from explicit keys or the Universe URL."""
    ref = {
        "workspace": source.get("workspace"),
        "project": source.get("project"),
        "version": source.get("version"),
    }
    url = source.get("url", "")
    if url and (not ref["workspace"] or not ref["project"]):
        parts = [p for p in urlparse(url).path.split("/") if p]
        if len(parts) >= 1 and not ref["workspace"]:
            ref["workspace"] = parts[0]
        if len(parts) >= 2 and not ref["project"]:
            ref["project"] = parts[1]
        if len(parts) >= 3 and not ref["version"]:
            ref["version"] = parts[2]
    return ref


def download_roboflow(source: Dict[str, Any], dest: Path) -> Dict[str, Any]:
    """Download a Roboflow Universe dataset export via the Roboflow REST API."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        return _result(source, "failed", dest,
                       error="ROBOFLOW_API_KEY environment variable is not set")

    ref = _parse_roboflow_ref(source)
    if not ref["workspace"] or not ref["project"]:
        return _result(
            source, "failed", dest,
            error=f"Cannot resolve Roboflow workspace/project from source config "
                  f"(url={source.get('url')!r}). Add explicit 'workspace' and "
                  f"'project' keys to the source entry.",
        )

    export_format = ROBOFLOW_FORMATS.get(source.get("format", "yolo"), "yolov8")
    api = "https://api.roboflow.com"
    try:
        version = ref["version"]
        if not version:
            resp = requests.get(f"{api}/{ref['workspace']}/{ref['project']}",
                                params={"api_key": api_key}, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            versions = resp.json().get("versions", [])
            if not versions:
                return _result(source, "failed", dest, error="Project has no published versions")
            # Version ids look like "workspace/project/3" — take the newest.
            version = max(int(v["id"].rsplit("/", 1)[-1]) for v in versions)

        # The export may be generated on demand; poll until a link is available.
        export_url = f"{api}/{ref['workspace']}/{ref['project']}/{version}/{export_format}"
        link = None
        for _ in range(30):
            resp = requests.get(export_url, params={"api_key": api_key}, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            link = payload.get("export", {}).get("link")
            if link:
                break
            time.sleep(5)
        if not link:
            return _result(source, "failed", dest, error="Roboflow export link never became ready")

        archive = download_file(link, dest / f"{ref['project']}_v{version}_{export_format}.zip")
        extract_archive(archive, dest)
        archive.unlink(missing_ok=True)
    except requests.RequestException as e:
        return _result(source, "failed", dest, error=f"Roboflow API error: {e}")
    except IOError as e:
        return _result(source, "failed", dest, error=str(e))
    return _result(source, "downloaded", dest, version=str(version), export_format=export_format)


def download_http(source: Dict[str, Any], dest: Path) -> Dict[str, Any]:
    """Download a direct HTTP(S) file; archives are extracted in place."""
    url = source.get("url")
    if not url:
        return _result(source, "failed", dest, error="Missing 'url' in source config")
    filename = source.get("filename") or Path(urlparse(url).path).name or "download.bin"
    try:
        archive = download_file(url, dest / filename)
        if archive.name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
            extract_archive(archive, dest)
            archive.unlink(missing_ok=True)
    except IOError as e:
        return _result(source, "failed", dest, error=str(e))
    return _result(source, "downloaded", dest, url=url)


def download_github(source: Dict[str, Any], dest: Path) -> Dict[str, Any]:
    """Shallow-clone a git repository."""
    url = source.get("url")
    if not url:
        return _result(source, "failed", dest, error="Missing 'url' in source config")
    if dest.exists() and any(dest.iterdir()):
        return _result(source, "skipped", dest, url=url)
    try:
        run_command(["git", "clone", "--depth", "1", url, str(dest)])
    except subprocess.CalledProcessError as e:
        return _result(source, "failed", dest, error=(e.stderr or str(e)).strip())
    except subprocess.TimeoutExpired:
        return _result(source, "failed", dest, error="git clone timed out")
    return _result(source, "downloaded", dest, url=url)


def download_huggingface(source: Dict[str, Any], dest: Path) -> Dict[str, Any]:
    """Download a Hugging Face Hub dataset and save it to disk."""
    dataset_id = source.get("dataset_id") or source.get("url")
    if not dataset_id:
        return _result(source, "failed", dest, error="Missing 'dataset_id' in source config")
    try:
        from datasets import load_dataset  # deferred: optional heavy dependency
    except ImportError:
        return _result(source, "failed", dest,
                       error="'datasets' library not installed (pip install datasets)")
    try:
        ds = load_dataset(dataset_id)
        dest.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(dest))
    except Exception as e:  # noqa: BLE001 — hub errors vary widely
        return _result(source, "failed", dest, error=f"Hugging Face download failed: {e}")
    return _result(source, "downloaded", dest, dataset_id=dataset_id)


def _fetch_coco_image(img: Dict[str, Any], out_dir: Path) -> bool:
    """Download one COCO image by its coco_url; returns True on success."""
    out_path = out_dir / img["file_name"]
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    try:
        download_file(img["coco_url"], out_path, retries=3)
        return True
    except IOError as e:
        logger.warning(f"COCO image failed: {img['file_name']}: {e}")
        return False


def download_coco_subset(source: Dict[str, Any], dest: Path,
                         max_images: Optional[int] = None) -> Dict[str, Any]:
    """
    Download a category-filtered subset of COCO 2017.

    Fetches the official instance annotations, keeps only the categories named
    in the source config (e.g. cow/horse/sheep), downloads just the matching
    images, and writes filtered annotation files:
        <dest>/images/<split>/*.jpg
        <dest>/annotations/instances_<split>_filtered.json
    """
    wanted = [c.lower() for c in source.get("classes", [])]
    if not wanted:
        return _result(source, "failed", dest, error="Missing 'classes' in source config")
    splits = source.get("splits", list(COCO_SPLITS))

    try:
        archive = download_file(COCO_ANNOTATIONS_URL, dest / "annotations_trainval2017.zip")
        if not (dest / "annotations" / "instances_val2017.json").exists():
            extract_archive(archive, dest)
    except IOError as e:
        return _result(source, "failed", dest, error=str(e))

    totals = {"images": 0, "annotations": 0, "failed_images": 0}
    for split in splits:
        ann_file = dest / "annotations" / f"instances_{split}.json"
        if not ann_file.exists():
            logger.warning(f"COCO annotations missing for split {split}, skipping")
            continue

        logger.info(f"Filtering COCO {split} for categories: {wanted}")
        with open(ann_file, "r") as f:
            coco = json.load(f)

        categories = [c for c in coco["categories"] if c["name"].lower() in wanted]
        cat_ids = {c["id"] for c in categories}
        annotations = [a for a in coco["annotations"] if a["category_id"] in cat_ids]
        image_ids = {a["image_id"] for a in annotations}
        images = [i for i in coco["images"] if i["id"] in image_ids]
        del coco  # free the full annotation payload before downloading

        if max_images is not None and len(images) > max_images:
            images = sorted(images, key=lambda i: i["id"])[:max_images]
            kept_ids = {i["id"] for i in images}
            annotations = [a for a in annotations if a["image_id"] in kept_ids]

        img_dir = dest / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {len(images)} COCO {split} images...")
        failed = 0
        with ThreadPoolExecutor(max_workers=COCO_IMAGE_WORKERS) as pool:
            futures = [pool.submit(_fetch_coco_image, img, img_dir) for img in images]
            for future in as_completed(futures):
                if not future.result():
                    failed += 1

        downloaded = [i for i in images if (img_dir / i["file_name"]).exists()]
        kept_ids = {i["id"] for i in downloaded}
        filtered = {
            "images": downloaded,
            "annotations": [a for a in annotations if a["image_id"] in kept_ids],
            "categories": categories,
        }
        out_file = dest / "annotations" / f"instances_{split}_filtered.json"
        with open(out_file, "w") as f:
            json.dump(filtered, f)

        totals["images"] += len(downloaded)
        totals["annotations"] += len(filtered["annotations"])
        totals["failed_images"] += failed
        logger.info(f"COCO {split}: {len(downloaded)} images, "
                    f"{len(filtered['annotations'])} annotations ({failed} failed)")

    status = "downloaded" if totals["images"] > 0 else "failed"
    error = None if totals["images"] > 0 else "No COCO images downloaded"
    return _result(source, status, dest, error=error, **totals)


DOWNLOADERS = {
    "kaggle": download_kaggle,
    "roboflow": download_roboflow,
    "coco": download_coco_subset,
    "http": download_http,
    "github": download_github,
    "huggingface": download_huggingface,
}


# ─────────────────────────────── orchestration ───────────────────────────────


def download_all(config: Dict[str, Any],
                 only: Optional[List[str]] = None,
                 force: bool = False,
                 max_images: Optional[int] = None) -> List[Dict[str, Any]]:
    """Download all configured sources and write the manifest. Returns results."""
    raw_dir = BASE_DIR / config.get("paths", {}).get("raw_dir", "data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    sources = config.get("datasets", {}).get("sources", [])
    if only:
        unknown = set(only) - {s.get("name") for s in sources}
        if unknown:
            raise ValueError(f"Unknown source name(s): {sorted(unknown)}")
        sources = [s for s in sources if s.get("name") in only]

    results: List[Dict[str, Any]] = []
    for source in sources:
        name = source.get("name", "unnamed")
        source_type = source.get("type", "unknown")
        dest = raw_dir / name
        handler = DOWNLOADERS.get(source_type)

        logger.info("-" * 60)
        logger.info(f"Source: {name} (type={source_type})")

        if handler is None:
            results.append(_result(source, "failed", dest,
                                   error=f"Unsupported source type: {source_type}"))
            continue

        if force and dest.exists():
            logger.info(f"--force: removing existing {dest}")
            shutil.rmtree(dest)
        elif not force and source_type != "coco" and count_files(dest) > 0:
            logger.info(f"Files already present in {dest}, skipping (use --force to re-download)")
            results.append(_result(source, "skipped", dest))
            continue

        if source_type == "coco":
            entry = handler(source, dest, max_images=max_images)
        else:
            entry = handler(source, dest)
        results.append(entry)
        logger.info(f"Result: {entry['status']} ({entry['file_count']} files)"
                    + (f" — {entry.get('error')}" if entry.get("error") else ""))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "sources": results,
    }
    manifest_path = raw_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("-" * 60)
    logger.info(f"Manifest written to {manifest_path}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Download configured dataset sources into data/raw/")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Path to model_config.yaml")
    parser.add_argument("--source", action="append", dest="sources",
                        help="Download only this named source (repeatable)")
    parser.add_argument("--list", action="store_true", help="List configured sources and exit")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap the number of images pulled per COCO split")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(BASE_DIR / config.get("paths", {}).get("logs_dir", "logs"))

    if args.list:
        for s in config.get("datasets", {}).get("sources", []):
            print(f"{s.get('name'):<30} type={s.get('type'):<12} task={s.get('task', '-')}")
        return 0

    results = download_all(config, only=args.sources, force=args.force, max_images=args.max_images)
    failed = [r for r in results if r["status"] == "failed"]
    for r in failed:
        logger.error(f"FAILED: {r['name']}: {r.get('error')}")
    return 1 if failed and len(failed) == len(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
