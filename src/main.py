import json
import random
from pathlib import Path

import cv2

from process import load_bisenet_model, load_midas_model, compute_depth_map, normalize_depth_map, synthesize


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_image_paths(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_sign_assets(sign_paths, signs_root):
    assets = []
    for path in sign_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[warn] Skip sign {path} (failed to load)")
            continue
        rel = path.relative_to(signs_root)
        assets.append({
            "image": img,
            "category": path.parent.name,
            "source": str(rel),
        })
    return assets


def main():
    base_dir = Path(__file__).resolve().parent
    bg_dir = (base_dir / "../data/not_synthesized").resolve()
    signs_dir = (base_dir / "../data/signs").resolve()
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
    for bg_path in background_paths:
        bg = cv2.imread(str(bg_path))
        if bg is None:
            print(f"[warn] Skip background {bg_path} (failed to load)")
            continue

        depth_map = compute_depth_map(bg, midas_model, midas_transform, midas_device)
        depth_norm = normalize_depth_map(depth_map)

        num_signs = random.randint(2, 5)
        composite, placements = synthesize(bg.copy(), sign_assets, bisenet_model, depth_norm, n_objects=num_signs)

        out_name = f"{bg_path.stem}_synthetic.png"
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

        num_composites += 1

    print(f"Saved {num_composites} composites to {output_dir}")
    print("Wrote per-image annotations beside each composite")


if __name__ == "__main__":
    main()
