from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib as mpl
import config
from matplotlib import font_manager
from ultralytics import YOLO

IMAGE_EXTS = config.YOLO_IMAGE_EXTS
SYNTH_DIR = config.YOLO_SYNTH_DIR
DATASET_DIR = config.YOLO_DATASET_DIR
MODEL_NAME = config.YOLO_MODEL_NAME
EPOCHS = config.YOLO_EPOCHS
BATCH_SIZE = config.YOLO_BATCH_SIZE
IMG_SIZE = config.YOLO_IMG_SIZE
TRAIN_RATIO = config.YOLO_TRAIN_RATIO
VAL_RATIO = config.YOLO_VAL_RATIO
RANDOM_SEED = config.YOLO_RANDOM_SEED
MIN_BOX_EDGE = config.YOLO_MIN_BOX_EDGE
MIN_BOX_AREA = config.YOLO_MIN_BOX_AREA

FONT_PATH = config.FONT_PATH
SRC_DETECT_DIR = config.YOLO_SRC_DETECT_DIR
DATA_RUNS_DIR = config.YOLO_DATA_RUNS_DIR

HARD_NEGATIVE_POOL_ENABLED = config.YOLO_HARD_NEGATIVE_POOL_ENABLED
HARD_NEGATIVE_POOL_DIR = config.YOLO_HARD_NEGATIVE_POOL_DIR
HARD_NEG_INCLUDE_UNLABELED_BACKGROUND = config.YOLO_HARD_NEG_INCLUDE_UNLABELED_BACKGROUND
HARD_NEG_UNLABELED_BACKGROUND_DIR = config.YOLO_HARD_NEG_UNLABELED_BACKGROUND_DIR
HARD_NEG_CONF_THRESHOLD = config.YOLO_HARD_NEG_CONF_THRESHOLD
HARD_NEG_MAX_IOU_WITH_GT = config.YOLO_HARD_NEG_MAX_IOU_WITH_GT
HARD_NEG_MIN_CROP_EDGE = config.YOLO_HARD_NEG_MIN_CROP_EDGE
HARD_NEG_MIN_CROP_AREA = config.YOLO_HARD_NEG_MIN_CROP_AREA
HARD_NEG_PREDICT_BATCH_SIZE = config.YOLO_HARD_NEG_PREDICT_BATCH_SIZE
HARD_NEG_FALLBACK_TO_CPU_ON_OOM = config.YOLO_HARD_NEG_FALLBACK_TO_CPU_ON_OOM

HARD_NEG_AVAILABLE = True
HARD_NEG_IMPORT_ERROR = "MEOW"

try:
    from hard_negative_pool import (
        HardNegativeConfig,
        inject_pool_into_train_split,
        mine_and_store_hard_negatives,
        resolve_train_weights,
    )

    HARD_NEG_AVAILABLE = True
except Exception as exc:  # pragma: no cover - fallback path for optional module
    HARD_NEG_IMPORT_ERROR = str(exc)


@dataclass
class Sample:
    image_path: Path
    ann_path: Path
    signs: List[dict]
    group_id: str


def configure_matplotlib_fonts() -> None:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Font file not found: {FONT_PATH}")
    font_manager.fontManager.addfont(str(FONT_PATH))
    font_prop = font_manager.FontProperties(fname=str(FONT_PATH))
    font_name = font_prop.get_name()
    mpl.rcParams["font.family"] = [font_name]
    mpl.rcParams["font.sans-serif"] = [font_name]
    mpl.rcParams["axes.unicode_minus"] = False


def discover_samples(synth_dir: Path) -> List[Sample]:
    samples: List[Sample] = []
    for img_path in synth_dir.rglob("*"):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        ann_path = img_path.with_suffix(".json")
        if not ann_path.exists():
            continue
        with ann_path.open("r", encoding="utf-8") as ann_file:
            ann = json.load(ann_file)
        signs = ann.get("signs", [])
        background = ann.get("background", "")
        background_path = Path(background)
        if len(background_path.parts) > 1:
            group_id = background_path.parts[0]
        else:
            flat_name = background_path.stem if background else img_path.stem
            group_id = flat_name.split("_frame_")[0].rsplit("_", 1)[0]
        samples.append(Sample(image_path=img_path, ann_path=ann_path, signs=signs, group_id=group_id))
    return samples


def build_class_map(samples: Sequence[Sample]) -> Dict[str, int]:
    categories = set()
    for sample in samples:
        for sign in sample.signs:
            label = sign.get("category")
            if label:
                categories.add(label)
    if not categories:
        raise RuntimeError("No categories detected in annotations.")
    return {name: idx for idx, name in enumerate(sorted(categories))}


def _group_class_histogram(samples: Sequence[Sample]) -> Counter:
    counts: Counter = Counter()
    for sample in samples:
        for sign in sample.signs:
            label = sign.get("category")
            if label:
                counts[label] += 1
    return counts


def split_samples_three(
    samples: Sequence[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    n = len(samples)
    if n < 3:
        raise RuntimeError("Need at least three annotated samples to perform train/val/test split.")

    groups: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id].append(sample)

    grouped_items = list(groups.items())
    if len(grouped_items) < 3:
        raise RuntimeError("Need synthesized samples from at least three source groups/videos for group-aware splitting.")

    rng = random.Random(seed)
    rng.shuffle(grouped_items)
    grouped_items.sort(key=lambda item: len(item[1]), reverse=True)

    target_train = n * train_ratio
    target_val = n * val_ratio
    splits: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}
    split_order = ("train", "val", "test")

    for idx, (_, group_samples) in enumerate(grouped_items):
        if idx < len(split_order):
            splits[split_order[idx]].extend(group_samples)
            continue
        train_gap = target_train - len(splits["train"])
        val_gap = target_val - len(splits["val"])
        if train_gap >= val_gap and train_gap > 0:
            target_split = "train"
        elif val_gap > 0:
            target_split = "val"
        else:
            target_split = "test"
        splits[target_split].extend(group_samples)

    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits["test"]
    return train_samples, val_samples, test_samples


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def is_valid_bbox(bbox: Sequence[float]) -> bool:
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    width = max(float(x2) - float(x1), 0.0)
    height = max(float(y2) - float(y1), 0.0)
    return width >= MIN_BOX_EDGE and height >= MIN_BOX_EDGE and width * height >= MIN_BOX_AREA


def write_label_file(label_path: Path, signs: Sequence[dict], class_map: Dict[str, int], width: int, height: int) -> None:
    lines: List[str] = []
    for sign in signs:
        label = sign.get("category")
        bbox = sign.get("bbox")
        if label not in class_map or not bbox or not is_valid_bbox(bbox):
            continue
        x1, y1, x2, y2 = bbox
        x1, x2 = sorted([float(x1), float(x2)])
        y1, y2 = sorted([float(y1), float(y2)])
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        cx /= max(width, 1)
        cy /= max(height, 1)
        bw /= max(width, 1)
        bh /= max(height, 1)
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 1e-6), 1.0)
        bh = min(max(bh, 1e-6), 1.0)
        class_id = class_map[label]
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def export_split(samples: Sequence[Sample], split_name: str, dataset_dir: Path, class_map: Dict[str, int]) -> None:
    img_dir = dataset_dir / split_name / "images"
    lbl_dir = dataset_dir / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        dest_img = img_dir / sample.image_path.name
        shutil.copy2(sample.image_path, dest_img)
        img = cv2.imread(str(dest_img))
        if img is None:
            raise RuntimeError(f"Failed to load copied image {dest_img}")
        label_path = lbl_dir / f"{dest_img.stem}.txt"
        write_label_file(label_path, sample.signs, class_map, img.shape[1], img.shape[0])


def write_dataset_yaml(dataset_dir: Path, class_map: Dict[str, int]) -> Path:
    yaml_path = dataset_dir / "dataset.yaml"
    names = [""] * len(class_map)
    for label, idx in class_map.items():
        names[idx] = label
    yaml_lines = [
        f"path: {dataset_dir.resolve()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "names:",
    ]
    yaml_lines.extend([f"  - {name}" for name in names])
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return yaml_path


def prepare_src_detect_dir(src_detect_dir: Path) -> None:
    reset_dir(src_detect_dir)


def next_detect_archive_dir(data_runs_dir: Path) -> Path:
    max_idx = 0
    for path in data_runs_dir.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if name == "detect":
            max_idx = max(max_idx, 1)
            continue
        if not name.startswith("detect"):
            continue
        suffix = name[len("detect") :]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    next_idx = max_idx + 1
    next_name = "detect" if next_idx == 1 else f"detect{next_idx}"
    return data_runs_dir / next_name


def archive_detect_run(src_detect_dir: Path, data_runs_dir: Path) -> Path:
    data_runs_dir.mkdir(parents=True, exist_ok=True)
    target_dir = next_detect_archive_dir(data_runs_dir)
    shutil.copytree(src_detect_dir, target_dir)
    return target_dir


def train_and_evaluate(
    data_yaml: Path,
    model_name: str,
    epochs: int,
    batch: int,
    imgsz: int,
    src_detect_dir: Path,
) -> None:
    model = YOLO(model_name)
    project = str(src_detect_dir)
    model.train(data=str(data_yaml), epochs=epochs, batch=batch, imgsz=imgsz, project=project, name="train", exist_ok=True)
    model.val(data=str(data_yaml), split="val", project=project, name="val", exist_ok=True)
    model.val(data=str(data_yaml), split="test", project=project, name="test", exist_ok=True)


def build_hard_negative_config() -> Optional["HardNegativeConfig"]:
    if not HARD_NEGATIVE_POOL_ENABLED:
        return None
    if not HARD_NEG_AVAILABLE:
        print(f"[hard-negative] disabled (import failed): {HARD_NEG_IMPORT_ERROR}")
        return None
    return HardNegativeConfig(
        pool_dir=HARD_NEGATIVE_POOL_DIR,
        mine_split="train",
        include_unlabeled_background=HARD_NEG_INCLUDE_UNLABELED_BACKGROUND,
        unlabeled_background_dir=HARD_NEG_UNLABELED_BACKGROUND_DIR,
        conf_threshold=HARD_NEG_CONF_THRESHOLD,
        max_iou_with_gt=HARD_NEG_MAX_IOU_WITH_GT,
        min_crop_edge=HARD_NEG_MIN_CROP_EDGE,
        min_crop_area=HARD_NEG_MIN_CROP_AREA,
        predict_batch_size=HARD_NEG_PREDICT_BATCH_SIZE,
        fallback_to_cpu_on_oom=HARD_NEG_FALLBACK_TO_CPU_ON_OOM,
    )


def main() -> None:
    synth_dir = SYNTH_DIR
    dataset_dir = DATASET_DIR
    if not synth_dir.exists():
        raise FileNotFoundError(f"Synthesized data directory not found: {synth_dir}")
    samples = discover_samples(synth_dir)
    if len(samples) < 3:
        raise RuntimeError("Need at least three annotated samples to perform train/val/test split.")
    class_map = build_class_map(samples)
    train_samples, val_samples, test_samples = split_samples_three(samples, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
    for split_name, split_samples in (
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ):
        split_hist = _group_class_histogram(split_samples)
        print(
            f"{split_name}: images={len(split_samples)} "
            f"boxes={sum(split_hist.values())} classes={len(split_hist)}"
        )
    reset_dir(dataset_dir)
    export_split(train_samples, "train", dataset_dir, class_map)
    export_split(val_samples, "val", dataset_dir, class_map)
    export_split(test_samples, "test", dataset_dir, class_map)
    hard_neg_config = build_hard_negative_config()
    if hard_neg_config is not None:
        reused_hard_negs = inject_pool_into_train_split(hard_neg_config.pool_dir, dataset_dir)
        print(f"[hard-negative] reused pooled negatives: {reused_hard_negs}")
    configure_matplotlib_fonts()
    data_yaml = write_dataset_yaml(dataset_dir, class_map)
    prepare_src_detect_dir(SRC_DETECT_DIR)
    train_and_evaluate(data_yaml, MODEL_NAME, EPOCHS, BATCH_SIZE, IMG_SIZE, SRC_DETECT_DIR)
    if hard_neg_config is not None:
        detector_weights = resolve_train_weights(SRC_DETECT_DIR, run_name="train")
        mined_hard_negs = mine_and_store_hard_negatives(
            weights_path=detector_weights,
            dataset_dir=dataset_dir,
            class_map=class_map,
            config=hard_neg_config,
            imgsz=IMG_SIZE,
        )
        print(
            "[hard-negative] newly mined negatives: "
            f"{mined_hard_negs} (saved to {hard_neg_config.pool_dir})"
        )
    archive_dir = archive_detect_run(SRC_DETECT_DIR, DATA_RUNS_DIR)
    print(f"Archived detect run to {archive_dir}")


if __name__ == "__main__":
    main()
