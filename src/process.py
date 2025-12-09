import cv2
import numpy as np
import random
import torch
import torch.nn.functional as F

from semantic_segmentation_bisenet_yolov8seg import load_bisenet_model, segment_with_bisenet, mask_to_color

def load_midas_model(model_type="DPT_Large"):
    """Load MiDaS depth model and matching transform."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas_model = torch.hub.load("isl-org/MiDaS", model_type)
    midas_model.to(device)
    midas_model.eval()

    transforms = torch.hub.load("isl-org/MiDaS", "transforms")
    transform = transforms.dpt_transform
    return midas_model, transform, device


def compute_depth_map(img_bgr, midas_model, midas_transform, device):
    """Return relative depth map from single RGB image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = midas_transform(img_rgb).to(device)

    with torch.no_grad():
        pred = midas_model(inp)
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        depth = pred.squeeze().cpu().numpy()
    return depth


def normalize_depth_map(depth_map, eps=1e-8):
    """Normalize arbitrary depth map to [0, 1] range."""
    depth = depth_map.astype(np.float32)
    min_val = np.min(depth)
    max_val = np.max(depth)
    spread = max(max_val - min_val, eps)
    return (depth - min_val) / spread


def depth_to_heatmap(depth_map):
    """Convert normalized depth map into a color heatmap for visualization."""
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)


def mean_std_match(fg, bg_patch):
    """Match foreground mean/std to background patch.
    fg: foreground image
    bg_patch: background region
    """
    fg = fg.astype(np.float32)
    bg = bg_patch.astype(np.float32)

    f_mean, f_std = fg.mean(), fg.std()
    b_mean, b_std = bg.mean(), bg.std()

    if f_std < 1: f_std = 1
    if b_std < 1: b_std = 1

    out = (fg - f_mean) * (b_std / f_std) + b_mean
    return np.clip(out, 0, 255).astype(np.uint8)


def random_warp(img):
    """Apply small affine + perspective jitter.
    img: foreground image
    """
    h, w = img.shape[:2]

    pts1 = np.float32([[0,0], [w,0], [0,h]])
    pts2 = np.float32([
        [random.uniform(0, w*0.05), random.uniform(0, h*0.05)],
        [w - random.uniform(0, w*0.05), random.uniform(0, h*0.05)],
        [random.uniform(0, w*0.05), h - random.uniform(0, h*0.05)]
    ])

    M = cv2.getAffineTransform(pts1, pts2)
    warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    jitter = 0.03
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts2 = np.float32([
        [w*random.uniform(0,jitter),     h*random.uniform(0,jitter)],
        [w*(1-random.uniform(0,jitter)), h*random.uniform(0,jitter)],
        [w*random.uniform(0,jitter),     h*(1-random.uniform(0,jitter))],
        [w*(1-random.uniform(0,jitter)), h*(1-random.uniform(0,jitter))]
    ])

    H = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(warped, H, (w, h), borderMode=cv2.BORDER_REFLECT)
    return warped


def make_soft_mask(fg, feather=5):
    """Create feathered alpha mask that tolerates very small inputs."""
    h, w = fg.shape[:2]
    if feather <= 0:
        return np.ones((h, w), dtype=np.float32)

    mask = np.ones((h, w), dtype=np.float32)
    v_feather = min(feather, h // 2) if h > 1 else 0
    h_feather = min(feather, w // 2) if w > 1 else 0

    if v_feather > 0:
        top_ramp = np.linspace(0, 1, v_feather, dtype=np.float32)
        mask[:v_feather, :] *= top_ramp[:, None]
        bottom_ramp = np.linspace(1, 0, v_feather, dtype=np.float32)
        mask[-v_feather:, :] *= bottom_ramp[:, None]

    if h_feather > 0:
        left_ramp = np.linspace(0, 1, h_feather, dtype=np.float32)
        mask[:, :h_feather] *= left_ramp[None, :]
        right_ramp = np.linspace(1, 0, h_feather, dtype=np.float32)
        mask[:, -h_feather:] *= right_ramp[None, :]

    blur = max(v_feather, h_feather)
    if blur > 0:
        k = blur * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    return mask


def paste_object(bg, fg, x, y):
    """Paste warped foreground onto background with alpha.
    bg: background image
    fg: warped foreground
    x: top-left x
    y: top-left y
    """
    h, w = fg.shape[:2]
    if x < 0 or y < 0 or x+w > bg.shape[1] or y+h > bg.shape[0]:
        return bg

    patch = bg[y:y+h, x:x+w]
    mask = make_soft_mask(fg)[...,None]
    out = patch * (1 - mask) + fg * mask
    bg[y:y+h, x:x+w] = out.astype(np.uint8)
    return bg


def boxes_overlap(box_a, box_b, padding=4):
    """Return True if two axis-aligned boxes overlap (optionally expanded)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ax1 -= padding
    ay1 -= padding
    ax2 += padding
    ay2 += padding
    bx1 -= padding
    by1 -= padding
    bx2 += padding
    by2 += padding

    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)
    return x_right > x_left and y_bottom > y_top


def synthesize(bg, fg_list, bisenet_model, depth_norm, n_objects=3):
    """Generate composite plus placement metadata using normalized depth.

    fg_list expects dictionaries containing at least `image`, `category`, and `source` keys.
    Returns (composited_image, placements_metadata).
    """
    h, w = bg.shape[:2]
    if depth_norm.shape != (h, w):
        raise ValueError("Depth map must match background resolution.")

    parsing = segment_with_bisenet(bg, bisenet_model)
    cv2.imwrite("main_bisenet_seg_vis.png", mask_to_color(parsing))

    SKY = 23
    ROAD = 7

    placements = []
    attempts = 0
    max_attempts = n_objects * 25

    while len(placements) < n_objects and attempts < max_attempts:
        attempts += 1
        x = np.random.randint(20, w-20)
        y = np.random.randint(20, h-20)

        if parsing[y, x] in [SKY, ROAD]:
            continue

        depth_value = depth_norm[y, x]
        if not np.isfinite(depth_value) or depth_value <= 0:
            continue

        fg_asset = random.choice(fg_list)
        fg_raw = fg_asset["image"]
        fg_label = fg_asset.get("category", "unknown")
        fg_source = fg_asset.get("source")

        scale = float(depth_value)
        new_w = max(1, int(fg_raw.shape[1] * scale))
        new_h = max(1, int(fg_raw.shape[0] * scale))

        if new_w == 0 or new_h == 0:
            continue

        fg_scaled = cv2.resize(fg_raw, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x_tl = int(np.clip(x - new_w // 2, 0, max(w - new_w, 0)))
        y_tl = int(np.clip(y - new_h // 2, 0, max(h - new_h, 0)))
        bg_patch = bg[y_tl:y_tl+new_h, x_tl:x_tl+new_w]

        if bg_patch.shape[:2] != (new_h, new_w):
            continue

        candidate_box = (x_tl, y_tl, x_tl + new_w, y_tl + new_h)
        if any(boxes_overlap(candidate_box, tuple(p["bbox"])) for p in placements):
            continue

        fg_adj = mean_std_match(fg_scaled, bg_patch)
        fg_warped = random_warp(fg_adj)

        bg = paste_object(bg, fg_warped, x_tl, y_tl)
        placements.append({
            "bbox": [int(x_tl), int(y_tl), int(x_tl + new_w), int(y_tl + new_h)],
            "category": fg_label,
            "source": fg_source,
        })

    return bg, placements
