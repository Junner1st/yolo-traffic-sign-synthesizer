from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import config
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation

IMAGE_EXTS = config.CUTOUT_IMAGE_EXTS
INPUT_DIR = config.CUTOUT_INPUT_DIR
OUTPUT_DIR = config.CUTOUT_OUTPUT_DIR
MODEL_ID = config.CUTOUT_MODEL_ID
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFER_SIZE = config.CUTOUT_INFER_SIZE
MASK_THRESHOLD = config.CUTOUT_MASK_THRESHOLD
ALPHA_SOFT = config.CUTOUT_ALPHA_SOFT
BBOX_PAD = config.CUTOUT_BBOX_PAD


def list_image_paths(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and ".ipynb_checkpoints" not in path.parts
    )


def load_birefnet_model(model_id: str, device: str):
    model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model


def preprocess_pil(image: Image.Image, size: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    image = image.convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor


def resolve_logits(output) -> torch.Tensor:
    if hasattr(output, "logits") and torch.is_tensor(output.logits):
        return output.logits
    if isinstance(output, dict):
        for key in ("logits", "pred", "mask"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    if isinstance(output, (list, tuple)):
        for item in reversed(output):
            if torch.is_tensor(item):
                return item
    if torch.is_tensor(output):
        return output
    raise RuntimeError("Unable to resolve logits tensor from BiRefNet output.")


def infer_mask_prob(model, image: Image.Image, size: int, device: str) -> np.ndarray:
    src_w, src_h = image.size
    model_dtype = next(model.parameters()).dtype
    inp = preprocess_pil(image, size=size, device=device, dtype=model_dtype)

    with torch.no_grad():
        output = model(inp)
        logits = resolve_logits(output)
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        if logits.ndim != 4:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
        prob = logits.sigmoid()
        prob = F.interpolate(prob, size=(src_h, src_w), mode="bilinear", align_corners=False)
        prob = prob.squeeze().detach().cpu().numpy().astype(np.float32)
    return np.clip(prob, 0.0, 1.0)


def build_sign_rgba(
    rel_path: Path,
    image: Image.Image,
    prob: np.ndarray,
    output_dir: Path,
    threshold: float,
    alpha_soft: bool,
) -> None:
    rgb_arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask_u8 = (prob * 255.0).astype(np.uint8)
    mask_bin = (prob >= threshold).astype(np.uint8) * 255
    alpha = mask_u8 if alpha_soft else mask_bin

    ys, xs = np.where(mask_bin > 0)
    if xs.size == 0 or ys.size == 0:
        return

    x0 = max(int(xs.min()) - BBOX_PAD, 0)
    y0 = max(int(ys.min()) - BBOX_PAD, 0)
    x1 = min(int(xs.max()) + BBOX_PAD + 1, rgb_arr.shape[1])
    y1 = min(int(ys.max()) + BBOX_PAD + 1, rgb_arr.shape[0])

    sign_rgb = rgb_arr[y0:y1, x0:x1]
    sign_alpha = alpha[y0:y1, x0:x1]
    sign_rgba = np.dstack([sign_rgb, sign_alpha]).astype(np.uint8)

    sign_path = (output_dir / rel_path).with_suffix(".png")
    sign_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sign_rgba, mode="RGBA").save(sign_path)


def main() -> None:
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_paths(input_dir)
    if not image_paths:
        raise RuntimeError(f"No images found under {input_dir}")

    model = load_birefnet_model(MODEL_ID, DEVICE)

    for path in tqdm(image_paths, desc="BiRefNet extracting", unit="img"):
        image = Image.open(path).convert("RGB")
        prob = infer_mask_prob(model, image, size=INFER_SIZE, device=DEVICE)
        rel_path = path.relative_to(input_dir)
        build_sign_rgba(
            rel_path=rel_path,
            image=image,
            prob=prob,
            output_dir=output_dir,
            threshold=MASK_THRESHOLD,
            alpha_soft=ALPHA_SOFT,
        )

    print(f"Processed {len(image_paths)} images.")
    print(f"Saved separated signs under: {output_dir}")


if __name__ == "__main__":
    main()
