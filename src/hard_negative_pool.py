from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import config
import torch
from ultralytics import YOLO

IMAGE_EXTS = config.YOLO_IMAGE_EXTS
HARD_NEG_MINE_SPLIT = config.HARD_NEG_MINE_SPLIT
HARD_NEG_IMGSZ = config.HARD_NEG_IMGSZ
HARD_NEG_CONF_THRESHOLD = config.HARD_NEG_CONF_THRESHOLD
HARD_NEG_MAX_IOU_WITH_GT = config.HARD_NEG_MAX_IOU_WITH_GT
HARD_NEG_MIN_CROP_EDGE = config.HARD_NEG_MIN_CROP_EDGE
HARD_NEG_MIN_CROP_AREA = config.HARD_NEG_MIN_CROP_AREA
HARD_NEG_PREDICT_BATCH_SIZE = config.HARD_NEG_PREDICT_BATCH_SIZE
HARD_NEG_INCLUDE_UNLABELED_BACKGROUND = config.HARD_NEG_INCLUDE_UNLABELED_BACKGROUND
HARD_NEG_FALLBACK_TO_CPU_ON_OOM = config.HARD_NEG_FALLBACK_TO_CPU_ON_OOM


@dataclass(frozen=True)
class HardNegativeConfig:
    pool_dir: Path
    mine_split: str = "train"
    include_unlabeled_background: bool = False
    unlabeled_background_dir: Optional[Path] = None
    conf_threshold: float = 0.6
    max_iou_with_gt: float = 0.05
    min_crop_edge: int = 24
    min_crop_area: int = 24 * 24
    predict_batch_size: int = 1
    fallback_to_cpu_on_oom: bool = True


@dataclass(frozen=True)
class MiningCandidate:
    image_path: Path
    label_path: Optional[Path]
    source_kind: str


def _iter_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def _slugify(value: str) -> str:
    cleaned = []
    for ch in value.strip():
        if ch.isalnum():
            cleaned.append(ch.lower())
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "class"


def resolve_train_weights(src_detect_dir: Path, run_name: str = "train") -> Path:
    weights_dir = src_detect_dir / run_name / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    if best_pt.exists():
        return best_pt
    if last_pt.exists():
        return last_pt
    raise FileNotFoundError(f"No trained weights found under {weights_dir}")


def inject_pool_into_train_split(pool_dir: Path, dataset_dir: Path) -> int:
    train_img_dir = dataset_dir / "train" / "images"
    train_lbl_dir = dataset_dir / "train" / "labels"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    if not pool_dir.exists():
        return 0

    copied = 0
    for class_dir in sorted(path for path in pool_dir.iterdir() if path.is_dir()):
        class_img_dir = class_dir / "images"
        if not class_img_dir.exists():
            continue
        for src_img in _iter_images(class_img_dir):
            stem = f"hnpool_{class_dir.name}_{src_img.stem}_{copied:08d}"
            dst_img = train_img_dir / f"{stem}{src_img.suffix.lower()}"
            dst_lbl = train_lbl_dir / f"{stem}.txt"
            shutil.copy2(src_img, dst_img)
            dst_lbl.write_text("", encoding="utf-8")
            copied += 1
    return copied


def _parse_yolo_label_file(label_path: Path, width: int, height: int) -> List[Tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes: List[Tuple[float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx_s, cy_s, bw_s, bh_s = parts
        try:
            cx = float(cx_s)
            cy = float(cy_s)
            bw = float(bw_s)
            bh = float(bh_s)
        except ValueError:
            continue
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def _gather_candidates(dataset_dir: Path, config: HardNegativeConfig) -> List[MiningCandidate]:
    split_img_dir = dataset_dir / config.mine_split / "images"
    split_lbl_dir = dataset_dir / config.mine_split / "labels"
    candidates: List[MiningCandidate] = []
    for img_path in _iter_images(split_img_dir):
        if img_path.stem.startswith("hnpool_"):
            continue
        candidates.append(
            MiningCandidate(
                image_path=img_path,
                label_path=split_lbl_dir / f"{img_path.stem}.txt",
                source_kind=config.mine_split,
            )
        )
    if config.include_unlabeled_background and config.unlabeled_background_dir:
        for img_path in _iter_images(config.unlabeled_background_dir):
            candidates.append(MiningCandidate(image_path=img_path, label_path=None, source_kind="unlabeled"))
    return candidates


def _class_name(class_map: Dict[str, int], class_id: int, model_names: Dict[int, str]) -> str:
    inv = {idx: name for name, idx in class_map.items()}
    return inv.get(class_id, model_names.get(class_id, f"class_{class_id}"))


def mine_and_store_hard_negatives(
    weights_path: Path,
    dataset_dir: Path,
    class_map: Dict[str, int],
    config: HardNegativeConfig,
    imgsz: int,
) -> int:
    candidates = _gather_candidates(dataset_dir, config)
    if not candidates:
        return 0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = YOLO(str(weights_path))
    model_names_raw = model.names if isinstance(model.names, dict) else {}
    model_names = {int(k): str(v) for k, v in model_names_raw.items()}

    config.pool_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.pool_dir / "manifest.jsonl"
    saved_count = 0
    base_stamp = int(time.time())

    with manifest_path.open("a", encoding="utf-8") as manifest_file:
        for candidate in candidates:
            try:
                preds = model.predict(
                    source=str(candidate.image_path),
                    conf=config.conf_threshold,
                    imgsz=imgsz,
                    stream=False,
                    verbose=False,
                    batch=max(config.predict_batch_size, 1),
                )
            except torch.OutOfMemoryError:
                if not config.fallback_to_cpu_on_oom:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                preds = model.predict(
                    source=str(candidate.image_path),
                    conf=config.conf_threshold,
                    imgsz=imgsz,
                    stream=False,
                    verbose=False,
                    batch=1,
                    device="cpu",
                )
            if not preds:
                continue
            result = preds[0]
            orig = result.orig_img
            if orig is None:
                continue
            height, width = orig.shape[:2]
            gt_boxes = []
            if candidate.label_path is not None:
                gt_boxes = _parse_yolo_label_file(candidate.label_path, width, height)
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for idx in range(len(boxes)):
                conf = float(boxes.conf[idx].item())
                cls_id = int(boxes.cls[idx].item())
                x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[idx].tolist()]
                if gt_boxes:
                    overlap = max((_bbox_iou((x1, y1, x2, y2), gt) for gt in gt_boxes), default=0.0)
                    if overlap >= config.max_iou_with_gt:
                        continue
                ix1 = max(int(x1), 0)
                iy1 = max(int(y1), 0)
                ix2 = min(int(x2), width)
                iy2 = min(int(y2), height)
                crop_w = ix2 - ix1
                crop_h = iy2 - iy1
                if crop_w < config.min_crop_edge or crop_h < config.min_crop_edge:
                    continue
                if crop_w * crop_h < config.min_crop_area:
                    continue
                crop = orig[iy1:iy2, ix1:ix2]
                if crop.size == 0:
                    continue

                class_name = _class_name(class_map, cls_id, model_names)
                class_dir_name = f"{cls_id:03d}_{_slugify(class_name)}"
                class_dir = config.pool_dir / class_dir_name
                images_dir = class_dir / "images"
                labels_dir = class_dir / "labels"
                images_dir.mkdir(parents=True, exist_ok=True)
                labels_dir.mkdir(parents=True, exist_ok=True)

                stem = f"hn_{base_stamp}_{saved_count:08d}"
                out_img = images_dir / f"{stem}.jpg"
                out_lbl = labels_dir / f"{stem}.txt"
                if not cv2.imwrite(str(out_img), crop):
                    continue
                out_lbl.write_text("", encoding="utf-8")
                saved_count += 1

                manifest_entry = {
                    "saved_image": str(out_img),
                    "saved_label": str(out_lbl),
                    "source_image": str(candidate.image_path),
                    "source_kind": candidate.source_kind,
                    "pred_class_id": cls_id,
                    "pred_class_name": class_name,
                    "confidence": conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "crop_xyxy": [ix1, iy1, ix2, iy2],
                }
                manifest_file.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
    return saved_count


def _load_class_map_from_dataset_yaml(dataset_dir: Path) -> Dict[str, int]:
    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.exists():
        return {}
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    names: List[str] = []
    in_names = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line == "names:":
            in_names = True
            continue
        if not in_names:
            continue
        if line.startswith("- "):
            names.append(line[2:].strip())
            continue
        break
    return {name: idx for idx, name in enumerate(names) if name}


def _default_paths() -> Dict[str, Path]:
    return {
        "dataset_dir": config.YOLO_DATASET_DIR,
        "pool_dir": config.YOLO_HARD_NEGATIVE_POOL_DIR,
        "unlabeled_background_dir": config.YOLO_HARD_NEG_UNLABELED_BACKGROUND_DIR,
        "src_detect_dir": config.YOLO_SRC_DETECT_DIR,
    }


def main() -> None:
    defaults = _default_paths()
    dataset_dir = defaults["dataset_dir"]
    pool_dir = defaults["pool_dir"]
    src_detect_dir = defaults["src_detect_dir"]
    unlabeled_background_dir = defaults["unlabeled_background_dir"]

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    weights_path = resolve_train_weights(src_detect_dir, run_name="train")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    class_map = _load_class_map_from_dataset_yaml(dataset_dir)
    config = HardNegativeConfig(
        pool_dir=pool_dir,
        mine_split=HARD_NEG_MINE_SPLIT,
        include_unlabeled_background=HARD_NEG_INCLUDE_UNLABELED_BACKGROUND,
        unlabeled_background_dir=unlabeled_background_dir,
        conf_threshold=HARD_NEG_CONF_THRESHOLD,
        max_iou_with_gt=HARD_NEG_MAX_IOU_WITH_GT,
        min_crop_edge=HARD_NEG_MIN_CROP_EDGE,
        min_crop_area=HARD_NEG_MIN_CROP_AREA,
        predict_batch_size=max(HARD_NEG_PREDICT_BATCH_SIZE, 1),
        fallback_to_cpu_on_oom=HARD_NEG_FALLBACK_TO_CPU_ON_OOM,
    )
    print(
        "[hard-negative-cli] settings: "
        f"split={config.mine_split} conf={config.conf_threshold} "
        f"max_iou={config.max_iou_with_gt} min_edge={config.min_crop_edge} "
        f"min_area={config.min_crop_area}"
    )
    mined = mine_and_store_hard_negatives(
        weights_path=weights_path,
        dataset_dir=dataset_dir,
        class_map=class_map,
        config=config,
        imgsz=HARD_NEG_IMGSZ,
    )
    print(f"[hard-negative-cli] mined={mined} pool_dir={pool_dir}")


if __name__ == "__main__":
    main()
