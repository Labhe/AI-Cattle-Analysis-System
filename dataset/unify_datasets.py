"""
Dataset Unification for the AI Cattle Analysis System (Phase 2).

Merges the heterogeneous raw datasets in ``data/raw/`` (downloaded by
``dataset/download_datasets.py``) into one canonical processed dataset under
``data/processed/``:

    data/processed/
      images/{train,val,test}/          detection images
      labels/{train,val,test}/          YOLO-format labels (unified class ids)
      masks/{train,val,test}/           binary masks rendered from COCO polygons
      classification/{train,val,test}/<label>/   breed / species image folders
      regression/{train,val,test}/      images with weight / BCS targets
      annotations.csv                   master index of every sample
      data.yaml                         Ultralytics dataset descriptor
      stats.json                        dataset statistics

Handled raw formats (auto-detected per source directory):
  - COCO instance JSON (including the ``*_filtered.json`` files this
    project's downloader produces)
  - YOLO datasets carrying a ``data.yaml``/``dataset.yaml`` with class names
  - Pascal VOC ``.xml`` sidecar annotations
  - Image-folder classification layouts (directory name = label)
  - CSV tables mapping images to weight / body-condition-score targets

Class ids are remapped onto the unified livestock classes from
``configs/model_config.yaml`` (``species.livestock_classes``). Corrupt images
are dropped and exact/near duplicates are removed across all sources via
perceptual hashing before the stratified train/val/test split.

Usage:
    python -m dataset.unify_datasets
    python -m dataset.unify_datasets --raw-dir data/raw --val-frac 0.15 --test-frac 0.15
"""

import argparse
import json
import logging
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import imagehash
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "model_config.yaml"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Directory names that never carry class information.
RESERVED_DIR_NAMES = {
    "images", "image", "img", "imgs", "labels", "label", "annotations", "masks",
    "train", "training", "val", "valid", "validation", "test", "testing",
    "data", "raw", "dataset", "datasets", "all", "export",
}

# Aliases mapped onto the unified livestock class names from the config.
CLASS_ALIASES = {
    "cattle": ["cattle", "cow", "cows", "bull", "bulls", "ox", "oxen",
               "calf", "calves", "heifer", "steer", "bovine"],
    "buffalo": ["buffalo", "buffaloes", "buffalos", "water buffalo"],
    "horse": ["horse", "horses", "mare", "stallion", "foal", "pony", "equine"],
    "sheep": ["sheep", "lamb", "lambs", "ewe", "ram"],
    "goat": ["goat", "goats", "kid goat", "billy goat", "caprine"],
    "pig": ["pig", "pigs", "swine", "hog", "hogs", "boar", "piglet", "sow"],
}

CSV_IMAGE_COLUMNS = ["image_path", "image", "image_name", "img_path", "filename",
                     "file_name", "file", "img", "path", "id"]
CSV_WEIGHT_COLUMNS = ["weight_kg", "weight", "live_weight", "body_weight", "weight_lbs"]
CSV_BCS_COLUMNS = ["body_condition_score", "bcs", "bcs_score", "condition_score"]

logger = logging.getLogger("dataset.unify")


@dataclass
class Record:
    """One unified sample gathered from a raw source."""
    image_path: Path                     # absolute path in data/raw
    source: str                          # raw source directory name
    task: str                            # 'detection' | 'classification' | 'regression'
    class_name: str                      # unified species, breed label, or 'unknown'
    class_id: Optional[int] = None       # unified livestock class id (detection)
    boxes: List[Tuple[int, float, float, float, float]] = field(default_factory=list)
    polygons: List[np.ndarray] = field(default_factory=list)  # absolute pixel coords
    image_size: Optional[Tuple[int, int]] = None              # (width, height)
    weight_kg: float = float("nan")
    bcs: float = float("nan")


# ─────────────────────────────── helpers ────────────────────────────────


def setup_logging(logs_dir: Path) -> None:
    """Configure console + file logging (idempotent)."""
    if logger.handlers:
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(logs_dir / "unify_datasets.log")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def load_class_map(config: Dict[str, Any]) -> Dict[str, int]:
    """Unified class name -> id from ``species.livestock_classes``."""
    classes = config.get("species", {}).get("livestock_classes", {})
    if not classes:
        raise ValueError("Config is missing species.livestock_classes")
    return {str(name).lower(): int(cid) for cid, name in classes.items()}


def build_alias_lookup(class_map: Dict[str, int]) -> Dict[str, int]:
    """Every known alias -> unified class id."""
    lookup: Dict[str, int] = {}
    for canonical, class_id in class_map.items():
        lookup[canonical] = class_id
        for alias in CLASS_ALIASES.get(canonical, []):
            lookup[alias] = class_id
    return lookup


def normalize_class_name(name: str, alias_lookup: Dict[str, int]) -> Optional[int]:
    """Map an arbitrary class string to a unified class id, or None if unknown."""
    return alias_lookup.get(re.sub(r"[\s_\-]+", " ", str(name).strip().lower()))


def canonical_name(class_id: Optional[int], alias_lookup: Dict[str, int],
                   default: str = "unknown") -> str:
    """Canonical class name (a CLASS_ALIASES key) for a unified class id."""
    return next((k for k, v in alias_lookup.items()
                 if v == class_id and k in CLASS_ALIASES), default)


def class_id_from_tokens(path: Path, source_dir: Path, alias_lookup: Dict[str, int]) -> Optional[int]:
    """
    Infer a species from directory-name tokens between the source root and the
    image (e.g. ``cid_cow/...`` -> cattle). Filenames are ignored: too noisy.
    """
    for part in reversed(path.relative_to(source_dir).parts[:-1] + (source_dir.name,)):
        for token in re.split(r"[\s_\-]+", part.lower()):
            if token in alias_lookup:
                return alias_lookup[token]
    return None


def is_valid_image(path: Path) -> Optional[Tuple[int, int]]:
    """Return (width, height) if the file is a readable image, else None."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.size
    except Exception:  # noqa: BLE001 — PIL raises many corrupt-file errors
        return None


def perceptual_hash(path: Path) -> Optional[str]:
    try:
        with Image.open(path) as img:
            return str(imagehash.phash(img))
    except Exception:  # noqa: BLE001
        return None


def voc_box_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float,
                    width: int, height: int) -> Tuple[float, float, float, float]:
    """Convert absolute corner coords to normalized YOLO (cx, cy, w, h)."""
    cx = (xmin + xmax) / 2.0 / width
    cy = (ymin + ymax) / 2.0 / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    clamp = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    return clamp(cx), clamp(cy), clamp(w), clamp(h)


# ───────────────────────────── format parsers ─────────────────────────────
# Each parser yields Records and adds every image it claimed to `consumed`
# so the fallback classification pass does not double-count it.


def parse_coco_files(source_dir: Path, alias_lookup: Dict[str, int],
                     consumed: Set[Path]) -> List[Record]:
    """Parse COCO instance JSON files found anywhere under the source."""
    records: List[Record] = []
    for json_file in sorted(source_dir.rglob("*.json")):
        try:
            with open(json_file, "r") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict) or "images" not in payload or "annotations" not in payload:
            continue

        cat_to_unified = {
            c["id"]: normalize_class_name(c["name"], alias_lookup)
            for c in payload.get("categories", [])
        }
        anns_by_image: Dict[int, List[dict]] = {}
        for ann in payload["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        # Images may sit next to the JSON or in split folders elsewhere in the source.
        name_index = {p.name: p for p in source_dir.rglob("*")
                      if p.suffix.lower() in IMAGE_EXTENSIONS}

        matched = 0
        for img_meta in payload["images"]:
            img_path = name_index.get(Path(img_meta["file_name"]).name)
            if img_path is None or img_path in consumed:
                continue
            width, height = img_meta.get("width"), img_meta.get("height")
            if not width or not height:
                continue

            boxes: List[Tuple[int, float, float, float, float]] = []
            polygons: List[np.ndarray] = []
            for ann in anns_by_image.get(img_meta["id"], []):
                class_id = cat_to_unified.get(ann["category_id"])
                if class_id is None:
                    continue
                x, y, w, h = ann["bbox"]
                boxes.append((class_id, *voc_box_to_yolo(x, y, x + w, y + h, width, height)))
                seg = ann.get("segmentation")
                if isinstance(seg, list):  # polygon segmentation only (not RLE)
                    for poly in seg:
                        if len(poly) >= 6:
                            polygons.append(np.array(poly, dtype=np.float32).reshape(-1, 2))
            if not boxes:
                continue

            primary = max((b[0] for b in boxes), key=[b[0] for b in boxes].count)
            records.append(Record(
                image_path=img_path, source=source_dir.name, task="detection",
                class_name=canonical_name(primary, alias_lookup, str(primary)),
                class_id=primary, boxes=boxes, polygons=polygons,
                image_size=(width, height),
            ))
            consumed.add(img_path)
            matched += 1
        if matched:
            logger.info(f"[{source_dir.name}] COCO {json_file.name}: {matched} images")
    return records


def _load_yolo_names(yaml_file: Path) -> Optional[Dict[int, str]]:
    """Read the class-name map from a YOLO data.yaml, if present."""
    try:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return None


def _find_yolo_label(img_path: Path) -> Optional[Path]:
    """Locate the YOLO .txt sidecar for an image (same dir or images->labels)."""
    sidecar = img_path.with_suffix(".txt")
    if sidecar.exists():
        return sidecar
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() in ("images", "image", "img", "imgs"):
            candidate = Path(*parts[:i], "labels", *parts[i + 1:-1],
                             img_path.stem + ".txt")
            if candidate.exists():
                return candidate
            break
    return None


def parse_yolo_datasets(source_dir: Path, alias_lookup: Dict[str, int],
                        consumed: Set[Path]) -> List[Record]:
    """Parse YOLO-format datasets that declare class names in a data.yaml."""
    records: List[Record] = []
    yaml_files = [p for p in sorted(source_dir.rglob("*.yaml")) + sorted(source_dir.rglob("*.yml"))
                  if p.name.lower() in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")]
    for yaml_file in yaml_files:
        names = _load_yolo_names(yaml_file)
        if names is None:
            continue
        local_to_unified = {lid: normalize_class_name(name, alias_lookup)
                            for lid, name in names.items()}
        dataset_root = yaml_file.parent

        matched = 0
        for img_path in sorted(dataset_root.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS or img_path in consumed:
                continue
            label_file = _find_yolo_label(img_path)
            if label_file is None:
                continue

            boxes: List[Tuple[int, float, float, float, float]] = []
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        local_id = int(float(parts[0]))
                        coords = [float(v) for v in parts[1:5]]
                    except ValueError:
                        continue
                    unified = local_to_unified.get(local_id)
                    if unified is None:
                        continue
                    boxes.append((unified, *coords))
            if not boxes:
                consumed.add(img_path)  # labelled but no livestock class kept
                continue

            primary = max((b[0] for b in boxes), key=[b[0] for b in boxes].count)
            records.append(Record(
                image_path=img_path, source=source_dir.name, task="detection",
                class_name=canonical_name(primary, alias_lookup, str(primary)),
                class_id=primary, boxes=boxes,
            ))
            consumed.add(img_path)
            matched += 1
        if matched:
            logger.info(f"[{source_dir.name}] YOLO {yaml_file.relative_to(source_dir)}: {matched} images")
    return records


def parse_voc_sidecars(source_dir: Path, alias_lookup: Dict[str, int],
                       consumed: Set[Path]) -> List[Record]:
    """Parse Pascal VOC .xml sidecar annotations."""
    records: List[Record] = []
    matched = 0
    for xml_file in sorted(source_dir.rglob("*.xml")):
        img_path = next((xml_file.with_suffix(ext) for ext in IMAGE_EXTENSIONS
                         if xml_file.with_suffix(ext).exists()), None)
        if img_path is None or img_path in consumed:
            continue
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            continue

        size = root.find("size")
        try:
            width = int(size.find("width").text)
            height = int(size.find("height").text)
        except (AttributeError, TypeError, ValueError):
            dims = is_valid_image(img_path)
            if dims is None:
                continue
            width, height = dims
        if width <= 0 or height <= 0:
            continue

        boxes: List[Tuple[int, float, float, float, float]] = []
        for obj in root.findall("object"):
            name_el, bnd = obj.find("name"), obj.find("bndbox")
            if name_el is None or bnd is None:
                continue
            class_id = normalize_class_name(name_el.text or "", alias_lookup)
            if class_id is None:
                continue
            try:
                corners = [float(bnd.find(tag).text) for tag in ("xmin", "ymin", "xmax", "ymax")]
            except (AttributeError, TypeError, ValueError):
                continue
            boxes.append((class_id, *voc_box_to_yolo(*corners, width, height)))
        if not boxes:
            continue

        primary = max((b[0] for b in boxes), key=[b[0] for b in boxes].count)
        records.append(Record(
            image_path=img_path, source=source_dir.name, task="detection",
            class_name=canonical_name(primary, alias_lookup, str(primary)),
            class_id=primary, boxes=boxes, image_size=(width, height),
        ))
        consumed.add(img_path)
        matched += 1
    if matched:
        logger.info(f"[{source_dir.name}] VOC sidecars: {matched} images")
    return records


def parse_csv_tables(source_dir: Path, alias_lookup: Dict[str, int],
                     consumed: Set[Path]) -> List[Record]:
    """Parse CSV tables mapping images to weight / BCS regression targets."""
    records: List[Record] = []
    for csv_file in sorted(source_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
        except Exception:  # noqa: BLE001 — malformed CSVs are simply skipped
            continue
        columns = {c.lower().strip(): c for c in df.columns}
        img_col = next((columns[c] for c in CSV_IMAGE_COLUMNS if c in columns), None)
        weight_col = next((columns[c] for c in CSV_WEIGHT_COLUMNS if c in columns), None)
        bcs_col = next((columns[c] for c in CSV_BCS_COLUMNS if c in columns), None)
        if img_col is None or (weight_col is None and bcs_col is None):
            continue

        name_index = {p.name: p for p in source_dir.rglob("*")
                      if p.suffix.lower() in IMAGE_EXTENSIONS}
        stem_index = {p.stem: p for p in name_index.values()}

        matched = 0
        for _, row in df.iterrows():
            ref = str(row[img_col]).strip()
            img_path = ((source_dir / ref) if (source_dir / ref).is_file() else None) \
                or (csv_file.parent / ref if (csv_file.parent / ref).is_file() else None) \
                or name_index.get(Path(ref).name) or stem_index.get(Path(ref).stem)
            if img_path is None or img_path in consumed:
                continue

            weight = pd.to_numeric(row[weight_col], errors="coerce") if weight_col else float("nan")
            bcs = pd.to_numeric(row[bcs_col], errors="coerce") if bcs_col else float("nan")
            if pd.isna(weight) and pd.isna(bcs):
                continue

            class_id = class_id_from_tokens(img_path, source_dir, alias_lookup)
            records.append(Record(
                image_path=img_path, source=source_dir.name, task="regression",
                class_name=canonical_name(class_id, alias_lookup),
                class_id=class_id, weight_kg=float(weight), bcs=float(bcs),
            ))
            consumed.add(img_path)
            matched += 1
        if matched:
            logger.info(f"[{source_dir.name}] CSV {csv_file.name}: {matched} regression samples")
    return records


def parse_classification_folders(source_dir: Path, alias_lookup: Dict[str, int],
                                 consumed: Set[Path]) -> Tuple[List[Record], int]:
    """
    Classify remaining images by their directory names.

    An image's immediate parent directory becomes its label (breed folders are
    kept verbatim); species aliases anywhere in the directory chain are used
    as fallback. Images with no label-bearing directory are skipped — no dummy
    labels are invented. Returns (records, skipped_count).
    """
    records: List[Record] = []
    skipped = 0
    for img_path in sorted(source_dir.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS or img_path in consumed:
            continue

        parent = img_path.parent.name
        parent_norm = re.sub(r"[\s_\-]+", " ", parent.strip().lower())
        species_id = normalize_class_name(parent, alias_lookup)

        if species_id is not None:
            label, class_id = canonical_name(species_id, alias_lookup), species_id
        elif parent_norm not in RESERVED_DIR_NAMES and img_path.parent != source_dir:
            label, class_id = parent.strip(), class_id_from_tokens(img_path, source_dir, alias_lookup)
        else:
            class_id = class_id_from_tokens(img_path, source_dir, alias_lookup)
            if class_id is None:
                skipped += 1
                continue
            label = canonical_name(class_id, alias_lookup)

        records.append(Record(
            image_path=img_path, source=source_dir.name, task="classification",
            class_name=label, class_id=class_id,
        ))
        consumed.add(img_path)
    if records:
        labels = sorted({r.class_name for r in records})
        logger.info(f"[{source_dir.name}] classification folders: {len(records)} images, "
                    f"{len(labels)} labels")
    return records, skipped


def collect_records(raw_dir: Path, alias_lookup: Dict[str, int]) -> Tuple[List[Record], int]:
    """Run every parser over every raw source directory."""
    records: List[Record] = []
    skipped_unlabeled = 0
    source_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    if not source_dirs:
        logger.warning(f"No source directories found under {raw_dir}")
    for source_dir in source_dirs:
        consumed: Set[Path] = set()
        records += parse_coco_files(source_dir, alias_lookup, consumed)
        records += parse_yolo_datasets(source_dir, alias_lookup, consumed)
        records += parse_voc_sidecars(source_dir, alias_lookup, consumed)
        records += parse_csv_tables(source_dir, alias_lookup, consumed)
        cls_records, skipped = parse_classification_folders(source_dir, alias_lookup, consumed)
        records += cls_records
        skipped_unlabeled += skipped
    return records, skipped_unlabeled


# ─────────────────────────── validation & splits ───────────────────────────


def validate_and_dedupe(records: List[Record]) -> Tuple[List[Record], Dict[str, int]]:
    """Drop corrupt images and near-duplicates (perceptual hash) across sources."""
    seen_hashes: Set[str] = set()
    kept: List[Record] = []
    dropped = {"corrupt": 0, "duplicate": 0}
    for record in tqdm(records, desc="Validating images"):
        dims = is_valid_image(record.image_path)
        if dims is None:
            dropped["corrupt"] += 1
            continue
        record.image_size = record.image_size or dims
        phash = perceptual_hash(record.image_path)
        if phash is None:
            dropped["corrupt"] += 1
            continue
        if phash in seen_hashes:
            dropped["duplicate"] += 1
            continue
        seen_hashes.add(phash)
        kept.append(record)
    return kept, dropped


def assign_splits(records: List[Record], val_frac: float, test_frac: float,
                  seed: int) -> Dict[str, List[Record]]:
    """Stratified train/val/test split, grouped per task."""
    splits: Dict[str, List[Record]] = {"train": [], "val": [], "test": []}
    holdout = val_frac + test_frac
    if not 0 < holdout < 1:
        raise ValueError("val_frac + test_frac must be in (0, 1)")

    for task in ("detection", "classification", "regression"):
        task_records = [r for r in records if r.task == task]
        if not task_records:
            continue
        if len(task_records) < 3:
            splits["train"] += task_records
            continue

        labels = [r.class_name for r in task_records]
        try:
            train, rest = train_test_split(task_records, test_size=holdout,
                                           stratify=labels, random_state=seed)
        except ValueError:  # a class has too few members to stratify
            train, rest = train_test_split(task_records, test_size=holdout,
                                           random_state=seed)
        if len(rest) < 2:
            val, test = rest, []
        else:
            try:
                val, test = train_test_split(rest, test_size=test_frac / holdout,
                                             stratify=[r.class_name for r in rest],
                                             random_state=seed)
            except ValueError:
                val, test = train_test_split(rest, test_size=test_frac / holdout,
                                             random_state=seed)
        splits["train"] += train
        splits["val"] += val
        splits["test"] += test
    return splits


# ────────────────────────────── output writers ──────────────────────────────


def render_mask(record: Record, mask_path: Path) -> bool:
    """Render a binary mask from COCO polygon segmentations."""
    if not record.polygons or record.image_size is None:
        return False
    width, height = record.image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    polys = [np.round(p).astype(np.int32) for p in record.polygons]
    cv2.fillPoly(mask, polys, 255)
    return bool(cv2.imwrite(str(mask_path), mask))


def write_outputs(splits: Dict[str, List[Record]], out_dir: Path,
                  class_map: Dict[str, int], max_side: Optional[int]) -> pd.DataFrame:
    """Copy images, write labels/masks, and return the master annotation table."""
    for split in splits:
        for sub in ("images", "labels", "masks"):
            (out_dir / sub / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "classification" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "regression" / split).mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    counter = 0
    for split, records in splits.items():
        for record in tqdm(records, desc=f"Writing {split}"):
            ext = record.image_path.suffix.lower()
            new_name = f"{record.source}_{counter:06d}{ext}"
            counter += 1

            if record.task == "detection":
                dest = out_dir / "images" / split / new_name
            elif record.task == "classification":
                label_dir = out_dir / "classification" / split / record.class_name.replace("/", "_")
                label_dir.mkdir(parents=True, exist_ok=True)
                dest = label_dir / new_name
            else:
                dest = out_dir / "regression" / split / new_name

            if max_side and record.image_size and max(record.image_size) > max_side:
                img = cv2.imread(str(record.image_path))
                if img is None:
                    continue
                scale = max_side / max(img.shape[:2])
                img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)),
                                 interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(dest), img):
                    continue
            else:
                shutil.copy2(record.image_path, dest)

            has_mask = False
            if record.task == "detection":
                label_path = out_dir / "labels" / split / (dest.stem + ".txt")
                with open(label_path, "w") as f:
                    for class_id, cx, cy, w, h in record.boxes:
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                has_mask = render_mask(record, out_dir / "masks" / split / (dest.stem + ".png"))

            rows.append({
                "image": str(dest.relative_to(out_dir)),
                "source_path": str(record.image_path),
                "split": split,
                "task": record.task,
                "source": record.source,
                "class_name": record.class_name,
                "class_id": record.class_id if record.class_id is not None else -1,
                "n_boxes": len(record.boxes),
                "has_mask": has_mask,
                "weight_kg": record.weight_kg,
                "body_condition_score": record.bcs,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "annotations.csv", index=False)

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_map),
        "names": {cid: name for name, cid in sorted(class_map.items(), key=lambda kv: kv[1])},
    }
    with open(out_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    return df


def write_stats(df: pd.DataFrame, dropped: Dict[str, int], skipped_unlabeled: int,
                out_dir: Path) -> Dict[str, Any]:
    """Write dataset statistics to stats.json and return them."""
    stats = {
        "total_samples": int(len(df)),
        "dropped_corrupt": dropped["corrupt"],
        "dropped_duplicates": dropped["duplicate"],
        "skipped_unlabeled": skipped_unlabeled,
        "by_split": df["split"].value_counts().to_dict() if len(df) else {},
        "by_task": df["task"].value_counts().to_dict() if len(df) else {},
        "by_source": df["source"].value_counts().to_dict() if len(df) else {},
        "by_class": df["class_name"].value_counts().to_dict() if len(df) else {},
        "regression_targets": {
            "with_weight": int(df["weight_kg"].notna().sum()) if len(df) else 0,
            "with_bcs": int(df["body_condition_score"].notna().sum()) if len(df) else 0,
        },
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


# ─────────────────────────────── orchestration ───────────────────────────────


def unify(raw_dir: Path, out_dir: Path, config: Dict[str, Any],
          val_frac: float = 0.15, test_frac: float = 0.15,
          seed: int = 42, max_side: Optional[int] = None) -> Dict[str, Any]:
    """Run the full unification pipeline; returns the stats dictionary."""
    class_map = load_class_map(config)
    alias_lookup = build_alias_lookup(class_map)

    logger.info(f"Scanning raw sources in {raw_dir}")
    records, skipped_unlabeled = collect_records(raw_dir, alias_lookup)
    logger.info(f"Collected {len(records)} candidate samples "
                f"({skipped_unlabeled} unlabeled images skipped)")

    records, dropped = validate_and_dedupe(records)
    logger.info(f"Kept {len(records)} samples after validation "
                f"({dropped['corrupt']} corrupt, {dropped['duplicate']} duplicates dropped)")

    out_dir.mkdir(parents=True, exist_ok=True)
    splits = assign_splits(records, val_frac, test_frac, seed)
    df = write_outputs(splits, out_dir, class_map, max_side)
    stats = write_stats(df, dropped, skipped_unlabeled, out_dir)

    logger.info(f"Unified dataset written to {out_dir}")
    logger.info(f"Splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Unify raw datasets into data/processed/")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Override paths.raw_dir from the config")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Override paths.processed_dir from the config")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-side", type=int, default=None,
                        help="Downscale images whose longest side exceeds this")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})
    setup_logging(BASE_DIR / paths.get("logs_dir", "logs"))

    raw_dir = args.raw_dir or BASE_DIR / paths.get("raw_dir", "data/raw")
    out_dir = args.out_dir or BASE_DIR / paths.get("processed_dir", "data/processed")
    if not raw_dir.exists():
        logger.error(f"Raw data directory does not exist: {raw_dir} "
                     f"(run dataset/download_datasets.py first)")
        return 1

    stats = unify(raw_dir, out_dir, config, val_frac=args.val_frac,
                  test_frac=args.test_frac, seed=args.seed, max_side=args.max_side)
    return 0 if stats["total_samples"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
