import json
import random
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

from process import (
    build_scene_context,
    compute_depth_map,
    load_bisenet_model,
    load_midas_model,
    normalize_depth_map,
    prepare_sign_asset,
    synthesize,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COMPOSITES_PER_BACKGROUND = 4
MIN_SIGNS_PER_IMAGE = 1
MAX_SIGNS_PER_IMAGE = 4


def list_image_paths(root: Path):
    ignore_token = ".ipynb_checkpoints"
    paths = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        if ignore_token in path.parts:
            continue
        paths.append(path)
    return sorted(paths)


def load_sign_assets(sign_paths, signs_root):
    assets = []
    for path in sign_paths:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        prepared = prepare_sign_asset(img)
        if prepared is None:
            print(f"[warn] Skip sign {path} (failed to load)")
            continue
        rel = path.relative_to(signs_root)
        assets.append(
            {
                "image": prepared["image"],
                "mask": prepared["mask"],
                "category": path.parent.name,
                "source": str(rel),
            }
        )
    return assets

def sample_sign_assets(sign_assets, n_objects):
    chosen = random.choices(sign_assets, k=n_objects)
    return list(chosen)

def initialize_sign_tracks(sign_assets, variant_idx, n_objects):
    tracks = []
    for track_idx, asset in enumerate(sample_sign_assets(sign_assets, n_objects)):
        tracks.append(
            {
                "track_id": f"v{variant_idx:02d}_t{track_idx:02d}",
                "asset": asset,
                "norm_x": None,
                "norm_y": None,
                "velocity_x": random.uniform(-0.01, 0.01),
                "velocity_y": random.uniform(-0.008, 0.008),
                "scale": None,
            }
        )
    return tracks


def main():
    base_dir = Path(__file__).resolve().parent
    bg_dir = (base_dir / "../data/not_synthesized").resolve()
    signs_dir = (base_dir / "../data/signs_cutout").resolve()
    output_dir = (base_dir / "../data/synthesized").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    assert bg_dir.exists(), f"Background directory not found: {bg_dir}"
    assert signs_dir.exists(), f"Sign directory not found: {signs_dir}"

    background_paths = list_image_paths(bg_dir)
    if not background_paths:
        raise RuntimeError(f"No background images detected under {bg_dir}")

    sign_paths = list_image_paths(signs_dir)
    sign_assets = load_sign_assets(sign_paths, signs_dir)
    if not sign_assets:
        raise RuntimeError(f"No valid sign images loaded from {signs_dir}")

    bisenet_model = load_bisenet_model()
    midas_model, midas_transform, midas_device = load_midas_model()

    num_composites = 0
    class_counts: Counter[str] = Counter()
    total_composites = len(background_paths) * COMPOSITES_PER_BACKGROUND
    with tqdm(total=total_composites, desc="Synthesizing", unit="image", dynamic_ncols=True) as progress_bar:
        for bg_path in background_paths:
            progress_bar.set_postfix_str(bg_path.name)

            bg = cv2.imread(str(bg_path))
            if bg is None:
                print(f"[warn] Skip background {bg_path} (failed to load)")
                progress_bar.update(COMPOSITES_PER_BACKGROUND)
                continue

            depth_map = compute_depth_map(bg, midas_model, midas_transform, midas_device)
            depth_norm = normalize_depth_map(depth_map)
            scene_context = build_scene_context(bg, bisenet_model, depth_norm)

            for variant_idx in range(COMPOSITES_PER_BACKGROUND):
                num_signs = random.randint(MIN_SIGNS_PER_IMAGE, MAX_SIGNS_PER_IMAGE)
                track_states = initialize_sign_tracks(sign_assets, variant_idx, num_signs)
                composite, placements, _ = synthesize(
                    bg,
                    track_states,
                    scene_context,
                )

                out_name = f"{bg_path.stem}_synthetic_v{variant_idx:02d}.png"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), composite)

                annotation = {
                    "background": str(bg_path.relative_to(bg_dir)),
                    "output": out_name,
                    "signs": placements,
                }
                ann_path = out_path.with_suffix(".json")
                with ann_path.open("w", encoding="utf-8") as ann_file:
                    json.dump(annotation, ann_file, indent=2, ensure_ascii=False)

                class_counts.update(
                    sign["category"]
                    for sign in placements
                    if sign.get("category")
                )
                num_composites += 1
                progress_bar.update(1)

    print(f"Saved {num_composites} composites to {output_dir}")
    print("Wrote per-image annotations beside each composite")
    print("Class counts:")
    if class_counts:
        for category, count in sorted(class_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {category}: {count}")
    else:
        print("  (no signs placed)")


if __name__ == "__main__":
    main()
