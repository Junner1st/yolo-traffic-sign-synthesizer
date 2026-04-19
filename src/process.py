import cv2
import numpy as np
import random
import torch
import torch.nn.functional as F
import config

from semantic_segmentation_bisenet_yolov8seg import load_bisenet_model, segment_with_bisenet


ROAD = config.PROCESS_ROAD
SIDEWALK = config.PROCESS_SIDEWALK
BUILDING = config.PROCESS_BUILDING
WALL = config.PROCESS_WALL
FENCE = config.PROCESS_FENCE
POLE = config.PROCESS_POLE
TRAFFIC_LIGHT = config.PROCESS_TRAFFIC_LIGHT
TRAFFIC_SIGN = config.PROCESS_TRAFFIC_SIGN
VEGETATION = config.PROCESS_VEGETATION
TERRAIN = config.PROCESS_TERRAIN
SKY = config.PROCESS_SKY
DYNAMIC_CLASSES = config.PROCESS_DYNAMIC_CLASSES

PLACEMENT_SUPPORT_CLASSES = config.PROCESS_PLACEMENT_SUPPORT_CLASSES
ANCHOR_CLASSES = config.PROCESS_ANCHOR_CLASSES
INVALID_PLACEMENT_CLASSES = config.PROCESS_INVALID_PLACEMENT_CLASSES

MIN_SCALE = config.PROCESS_MIN_SCALE
MAX_SCALE = config.PROCESS_MAX_SCALE
MIN_TARGET_SIDE = config.PROCESS_MIN_TARGET_SIDE
MIN_BBOX_EDGE = config.PROCESS_MIN_BBOX_EDGE
MIN_BBOX_AREA = config.PROCESS_MIN_BBOX_AREA
MAX_OBJECT_FRACTION = config.PROCESS_MAX_OBJECT_FRACTION
HARD_MASK_THRESHOLD = config.PROCESS_HARD_MASK_THRESHOLD
MOTION_BLUR_PROB = config.PROCESS_MOTION_BLUR_PROB
MOTION_BLUR_KERNEL_SIZES = config.PROCESS_MOTION_BLUR_KERNEL_SIZES
MOTION_BLUR_MAX_ANGLE = config.PROCESS_MOTION_BLUR_MAX_ANGLE


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


def prepare_sign_asset(raw_img):
    """Extract a tight foreground crop and alpha mask from a sign asset."""
    if raw_img is None:
        return None

    if raw_img.ndim == 2:
        color = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
        alpha = None
    elif raw_img.shape[2] == 4:
        color = raw_img[:, :, :3]
        alpha = raw_img[:, :, 3]
    else:
        color = raw_img[:, :, :3]
        alpha = None

    mask = extract_foreground_mask(color, alpha)
    if mask is None or float(mask.sum()) < 16:
        return None

    mask_u8 = (mask > 0.2).astype(np.uint8)
    if cv2.countNonZero(mask_u8) == 0:
        return None

    x, y, w, h = cv2.boundingRect(mask_u8)
    if w < 4 or h < 4:
        return None

    pad = 2
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, color.shape[1])
    y1 = min(y + h + pad, color.shape[0])

    cropped_image = color[y0:y1, x0:x1].copy()
    cropped_mask = soften_mask(mask[y0:y1, x0:x1], feather=4)
    if float(cropped_mask.sum()) < 16:
        return None

    return {"image": cropped_image, "mask": cropped_mask}


def extract_foreground_mask(color_img, alpha_channel=None):
    """Estimate foreground mask from alpha if present, otherwise use border-color heuristics."""
    if alpha_channel is not None:
        mask = alpha_channel.astype(np.float32) / 255.0
    else:
        h, w = color_img.shape[:2]
        border = np.concatenate(
            [
                color_img[0, :, :],
                color_img[-1, :, :],
                color_img[:, 0, :],
                color_img[:, -1, :],
            ],
            axis=0,
        ).astype(np.float32)
        bg_color = np.median(border, axis=0)
        color_dist = np.linalg.norm(color_img.astype(np.float32) - bg_color[None, None, :], axis=2)

        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0

        mask = (color_dist > 24).astype(np.float32)
        mask = np.maximum(mask, (sat > 32).astype(np.float32) * 0.7)
        if float(bg_color.mean()) > 180:
            mask = np.maximum(mask, (val < 245).astype(np.float32) * 0.5)
        if float(bg_color.mean()) < 70:
            mask = np.maximum(mask, (val > 28).astype(np.float32) * 0.5)
        mask = np.maximum(mask, edges)

    mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 18).astype(np.uint8), connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_idx = int(np.argmax(areas)) + 1
        if areas[best_idx - 1] > 0:
            mask_u8 = np.where(labels == best_idx, 255, 0).astype(np.uint8)

    if cv2.countNonZero(mask_u8) == 0:
        return np.ones(color_img.shape[:2], dtype=np.float32)

    return soften_mask(mask_u8.astype(np.float32) / 255.0, feather=3)


def soften_mask(mask, feather=3):
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if feather <= 0:
        return mask
    sigma = max(feather / 2.0, 0.5)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma), 0.0, 1.0)


def mean_std_match(fg, bg_patch, fg_mask=None):
    """Match masked foreground mean/std to the local background patch."""
    fg = fg.astype(np.float32)
    bg = bg_patch.astype(np.float32)

    if fg_mask is not None:
        mask = fg_mask > 0.15
        fg_pixels = fg[mask]
    else:
        fg_pixels = fg.reshape(-1, fg.shape[-1])

    bg_pixels = bg.reshape(-1, bg.shape[-1])
    if fg_pixels.size == 0 or bg_pixels.size == 0:
        return fg.astype(np.uint8)

    f_mean = fg_pixels.mean(axis=0)
    f_std = np.maximum(fg_pixels.std(axis=0), 1.0)
    b_mean = bg_pixels.mean(axis=0)
    b_std = np.maximum(bg_pixels.std(axis=0), 1.0)

    matched = (fg - f_mean) * (b_std / f_std) + b_mean
    matched = fg * 0.25 + matched * 0.75
    return np.clip(matched, 0, 255).astype(np.uint8)


def random_warp(img, mask):
    """Apply a shared affine + perspective warp to image and mask."""
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return img, mask

    affine_jitter = 0.05
    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    pts2 = np.float32(
        [
            [random.uniform(0, w * affine_jitter), random.uniform(0, h * affine_jitter)],
            [w - 1 - random.uniform(0, w * affine_jitter), random.uniform(0, h * affine_jitter)],
            [random.uniform(0, w * affine_jitter), h - 1 - random.uniform(0, h * affine_jitter)],
        ]
    )

    affine_mat = cv2.getAffineTransform(pts1, pts2)
    warped_img = cv2.warpAffine(
        img,
        affine_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask,
        affine_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    persp_jitter = 0.03
    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    pts2 = np.float32(
        [
            [w * random.uniform(0, persp_jitter), h * random.uniform(0, persp_jitter)],
            [w * (1 - random.uniform(0, persp_jitter)), h * random.uniform(0, persp_jitter)],
            [w * random.uniform(0, persp_jitter), h * (1 - random.uniform(0, persp_jitter))],
            [w * (1 - random.uniform(0, persp_jitter)), h * (1 - random.uniform(0, persp_jitter))],
        ]
    )

    persp_mat = cv2.getPerspectiveTransform(pts1, pts2)
    warped_img = cv2.warpPerspective(
        warped_img,
        persp_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpPerspective(
        warped_mask,
        persp_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_img, soften_mask(warped_mask, feather=2)


def boxes_overlap(box_a, box_b, padding=6):
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


def build_scene_context(bg, bisenet_model, depth_norm):
    """Create per-frame semantic priors for realistic sign placement."""
    h, w = bg.shape[:2]
    if depth_norm.shape != (h, w):
        raise ValueError("Depth map must match background resolution.")

    parsing = segment_with_bisenet(bg, bisenet_model)

    support_mask = np.isin(parsing, tuple(PLACEMENT_SUPPORT_CLASSES))
    blocked_mask = np.isin(parsing, tuple(INVALID_PLACEMENT_CLASSES))
    anchor_mask = np.isin(parsing, tuple(ANCHOR_CLASSES))
    road_mask = parsing == ROAD

    if not np.any(support_mask):
        support_mask = ~blocked_mask
    if not np.any(support_mask):
        support_mask = np.ones((h, w), dtype=bool)

    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    vertical_score = np.exp(-((ys - 0.42) / 0.22) ** 2)
    vertical_score *= ((ys > 0.10) & (ys < 0.82)).astype(np.float32)
    side_score = np.clip(np.abs(xs - 0.5) * 2.3, 0.22, 1.0)
    depth_score = 0.35 + 0.65 * np.clip(depth_norm, 0.0, 1.0)
    anchor_score = proximity_score(anchor_mask, scale=max(10.0, min(h, w) * 0.05))
    road_score = road_boundary_score(
        road_mask,
        target=max(12.0, min(h, w) * 0.07),
        sigma=max(10.0, min(h, w) * 0.08),
    )

    candidate_weights = support_mask.astype(np.float32)
    candidate_weights *= vertical_score
    candidate_weights *= side_score
    candidate_weights *= depth_score
    candidate_weights *= 0.40 + 0.60 * anchor_score
    candidate_weights *= 0.35 + 0.65 * road_score
    candidate_weights[blocked_mask] = 0.0
    candidate_weights = cv2.GaussianBlur(candidate_weights, (0, 0), sigmaX=max(1.2, min(h, w) * 0.01))

    if float(candidate_weights.sum()) <= 0:
        candidate_weights = (~blocked_mask).astype(np.float32)
    if float(candidate_weights.sum()) <= 0:
        candidate_weights = np.ones((h, w), dtype=np.float32)

    candidate_flat = candidate_weights.reshape(-1)
    candidate_cdf = np.cumsum(candidate_flat)

    return {
        "parsing": parsing,
        "depth_norm": depth_norm,
        "candidate_weights": candidate_weights,
        "candidate_cdf": candidate_cdf,
        "candidate_total": float(candidate_cdf[-1]),
    }


def proximity_score(mask, scale):
    if not np.any(mask):
        return np.full(mask.shape, 0.35, dtype=np.float32)
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    return np.exp(-dist / max(scale, 1.0)).astype(np.float32)


def road_boundary_score(road_mask, target, sigma):
    if not np.any(road_mask):
        return np.ones(road_mask.shape, dtype=np.float32)
    dist = cv2.distanceTransform((~road_mask).astype(np.uint8), cv2.DIST_L2, 3)
    score = np.exp(-((dist - target) ** 2) / (2.0 * sigma * sigma))
    return score.astype(np.float32)


def sample_weighted_index_from_cdf(cdf, total):
    if total <= 0 or cdf.size == 0:
        return None
    target = random.random() * total
    idx = int(np.searchsorted(cdf, target, side="left"))
    return min(idx, cdf.size - 1)


def sample_global_point(scene_context, margin):
    weights = scene_context["candidate_weights"]
    h, w = weights.shape
    for _ in range(12):
        idx = sample_weighted_index_from_cdf(scene_context["candidate_cdf"], scene_context["candidate_total"])
        if idx is None:
            return None
        y, x = divmod(idx, w)
        if margin <= x < w - margin and margin <= y < h - margin:
            return int(x), int(y)

    constrained = weights.copy()
    constrained[:margin, :] = 0.0
    constrained[h - margin :, :] = 0.0
    constrained[:, :margin] = 0.0
    constrained[:, w - margin :] = 0.0
    if float(constrained.sum()) <= 0:
        return None
    fallback_idx = int(np.argmax(constrained))
    y, x = divmod(fallback_idx, w)
    return int(x), int(y)


def sample_local_point(weight_patch, x_offset, y_offset):
    flat = weight_patch.reshape(-1)
    total = float(flat.sum())
    if total <= 0:
        return None
    cdf = np.cumsum(flat)
    idx = sample_weighted_index_from_cdf(cdf, total)
    if idx is None:
        return None
    y, x = divmod(idx, weight_patch.shape[1])
    return int(x + x_offset), int(y + y_offset)


def sample_track_point(track_state, scene_context):
    weights = scene_context["candidate_weights"]
    h, w = weights.shape
    margin = max(8, int(min(h, w) * 0.02))

    if track_state.get("norm_x") is None or track_state.get("norm_y") is None:
        return sample_global_point(scene_context, margin)

    prev_x = track_state["norm_x"] * max(w - 1, 1)
    prev_y = track_state["norm_y"] * max(h - 1, 1)
    vel_x = track_state.get("velocity_x", 0.0) * max(w - 1, 1)
    vel_y = track_state.get("velocity_y", 0.0) * max(h - 1, 1)
    pred_x = float(np.clip(prev_x + vel_x, margin, max(w - margin - 1, margin)))
    pred_y = float(np.clip(prev_y + vel_y, margin, max(h - margin - 1, margin)))

    for attempt in range(5):
        radius = int(max(20, min(h, w) * (0.05 + 0.02 * attempt)))
        x0 = max(int(pred_x) - radius, 0)
        y0 = max(int(pred_y) - radius, 0)
        x1 = min(int(pred_x) + radius + 1, w)
        y1 = min(int(pred_y) + radius + 1, h)
        local = weights[y0:y1, x0:x1].copy()
        if local.size == 0:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1]
        sigma = max(radius * 0.45, 6.0)
        local *= np.exp(-(((xx - pred_x) ** 2 + (yy - pred_y) ** 2) / (2.0 * sigma * sigma))).astype(np.float32)
        point = sample_local_point(local, x0, y0)
        if point is not None:
            return point

    return sample_global_point(scene_context, margin)


def depth_to_scale(depth_value):
    return float(np.clip(MIN_SCALE + depth_value * (MAX_SCALE - MIN_SCALE), MIN_SCALE, MAX_SCALE))


def local_depth_value(depth_norm, x, y, radius=2):
    h, w = depth_norm.shape
    x0 = max(x - radius, 0)
    y0 = max(y - radius, 0)
    x1 = min(x + radius + 1, w)
    y1 = min(y + radius + 1, h)
    patch = depth_norm[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    depth_value = float(np.nanmedian(patch))
    if not np.isfinite(depth_value):
        return None
    return float(np.clip(depth_value, 0.0, 1.0))


def bbox_from_mask(mask, x_offset, y_offset):
    mask_u8 = (mask > 0.25).astype(np.uint8)
    if cv2.countNonZero(mask_u8) == 0:
        return None
    x, y, w, h = cv2.boundingRect(mask_u8)
    return int(x + x_offset), int(y + y_offset), int(x + x_offset + w), int(y + y_offset + h)


def is_valid_bbox(box):
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 0)
    height = max(y2 - y1, 0)
    return width >= MIN_BBOX_EDGE and height >= MIN_BBOX_EDGE and width * height >= MIN_BBOX_AREA


def patch_is_plausible(parsing_patch, alpha_mask):
    visible = alpha_mask > 0.25
    if not np.any(visible):
        return False
    labels = parsing_patch[visible]
    invalid_ratio = float(np.isin(labels, tuple(INVALID_PLACEMENT_CLASSES)).mean())
    support_ratio = float(np.isin(labels, tuple(PLACEMENT_SUPPORT_CLASSES | ANCHOR_CLASSES)).mean())
    return invalid_ratio <= 0.08 and support_ratio >= 0.45


def blend_patch(bg_patch, fg_patch, alpha_mask):
    out = bg_patch.copy()
    hard_mask = alpha_mask >= HARD_MASK_THRESHOLD
    out[hard_mask] = fg_patch[hard_mask]
    return out


def apply_partial_occlusion(composite, base_bg, bbox):
    if random.random() >= 0.35:
        return composite

    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    if bw < MIN_BBOX_EDGE or bh < MIN_BBOX_EDGE:
        return composite

    orientation = "vertical" if bw <= bh or random.random() < 0.75 else "horizontal"
    if orientation == "vertical":
        occ_w = max(2, int(bw * random.uniform(0.12, 0.28)))
        occ_h = max(4, int(bh * random.uniform(0.75, 1.0)))
        occ_x = random.randint(x1, max(x1, x2 - occ_w))
        occ_y = random.randint(y1, max(y1, y2 - occ_h))
    else:
        occ_w = max(4, int(bw * random.uniform(0.45, 0.8)))
        occ_h = max(2, int(bh * random.uniform(0.10, 0.22)))
        occ_x = random.randint(x1, max(x1, x2 - occ_w))
        occ_y = random.randint(y1, max(y1, y2 - occ_h))

    patch_h = min(occ_h, composite.shape[0] - occ_y)
    patch_w = min(occ_w, composite.shape[1] - occ_x)
    if patch_h <= 0 or patch_w <= 0:
        return composite

    src_candidates = []
    if occ_x - patch_w - 2 >= 0:
        src_candidates.append((occ_x - patch_w - random.randint(0, 6), occ_y))
    if occ_x + patch_w + 2 < base_bg.shape[1]:
        src_candidates.append((occ_x + random.randint(2, 6), occ_y))
    if occ_y - patch_h - 2 >= 0:
        src_candidates.append((occ_x, occ_y - patch_h - random.randint(0, 6)))
    if occ_y + patch_h + 2 < base_bg.shape[0]:
        src_candidates.append((occ_x, occ_y + random.randint(2, 6)))

    src_patch = None
    random.shuffle(src_candidates)
    for src_x, src_y in src_candidates:
        src_x = int(np.clip(src_x, 0, max(base_bg.shape[1] - patch_w, 0)))
        src_y = int(np.clip(src_y, 0, max(base_bg.shape[0] - patch_h, 0)))
        candidate = base_bg[src_y:src_y + patch_h, src_x:src_x + patch_w]
        if candidate.shape[:2] == (patch_h, patch_w):
            src_patch = candidate.copy()
            break

    if src_patch is None:
        src_patch = np.full((patch_h, patch_w, 3), 96, dtype=np.uint8)

    alpha = np.full((patch_h, patch_w), random.uniform(0.75, 0.95), dtype=np.float32)
    alpha = soften_mask(alpha, feather=2)[:, :, None]
    dst = composite[occ_y:occ_y + patch_h, occ_x:occ_x + patch_w].astype(np.float32)
    out = dst * (1.0 - alpha) + src_patch.astype(np.float32) * alpha
    composite[occ_y:occ_y + patch_h, occ_x:occ_x + patch_w] = np.clip(out, 0, 255).astype(np.uint8)
    return composite


def update_track_state(track_state, center_x, center_y, scale, width, height):
    new_norm_x = center_x / max(width - 1, 1)
    new_norm_y = center_y / max(height - 1, 1)
    prev_norm_x = track_state.get("norm_x")
    prev_norm_y = track_state.get("norm_y")

    if prev_norm_x is not None and prev_norm_y is not None:
        delta_x = new_norm_x - prev_norm_x
        delta_y = new_norm_y - prev_norm_y
        track_state["velocity_x"] = float(np.clip(track_state.get("velocity_x", 0.0) * 0.6 + delta_x * 0.4, -0.03, 0.03))
        track_state["velocity_y"] = float(np.clip(track_state.get("velocity_y", 0.0) * 0.6 + delta_y * 0.4, -0.03, 0.03))
    else:
        track_state["velocity_x"] = float(track_state.get("velocity_x", 0.0) * 0.5)
        track_state["velocity_y"] = float(track_state.get("velocity_y", 0.0) * 0.5)

    track_state["norm_x"] = float(new_norm_x)
    track_state["norm_y"] = float(new_norm_y)
    track_state["scale"] = float(scale)


def decay_track_state(track_state):
    track_state["velocity_x"] = float(track_state.get("velocity_x", 0.0) * 0.5)
    track_state["velocity_y"] = float(track_state.get("velocity_y", 0.0) * 0.5)
    if track_state.get("norm_x") is not None:
        track_state["norm_x"] = float(np.clip(track_state["norm_x"] + track_state["velocity_x"], 0.0, 1.0))
    if track_state.get("norm_y") is not None:
        track_state["norm_y"] = float(np.clip(track_state["norm_y"] + track_state["velocity_y"], 0.0, 1.0))


def maybe_shift_from_pole(anchor_label, center_x, object_width, image_width):
    if anchor_label != POLE:
        return center_x
    if center_x < image_width * 0.5:
        return center_x + int(object_width * 0.22)
    return center_x - int(object_width * 0.22)


def place_track(composite, base_bg, track_state, scene_context, occupied_boxes):
    depth_norm = scene_context["depth_norm"]
    parsing = scene_context["parsing"]
    image_h, image_w = composite.shape[:2]
    asset = track_state["asset"]
    fg_base = asset["image"]
    mask_base = asset["mask"]

    for _ in range(12):
        point = sample_track_point(track_state, scene_context)
        if point is None:
            return None
        center_x, center_y = point

        depth_value = local_depth_value(depth_norm, center_x, center_y)
        if depth_value is None:
            continue

        target_scale = depth_to_scale(depth_value)
        prev_scale = track_state.get("scale")
        if prev_scale is not None:
            scale = float(np.clip(prev_scale * 0.75 + target_scale * 0.25, MIN_SCALE, MAX_SCALE))
        else:
            scale = target_scale

        raw_w = max(MIN_TARGET_SIDE, int(round(fg_base.shape[1] * scale)))
        raw_h = max(MIN_TARGET_SIDE, int(round(fg_base.shape[0] * scale)))
        raw_w = min(raw_w, max(int(image_w * MAX_OBJECT_FRACTION), MIN_TARGET_SIDE))
        raw_h = min(raw_h, max(int(image_h * MAX_OBJECT_FRACTION), MIN_TARGET_SIDE))
        if raw_w < MIN_TARGET_SIDE or raw_h < MIN_TARGET_SIDE:
            continue

        interp = cv2.INTER_AREA if raw_w < fg_base.shape[1] or raw_h < fg_base.shape[0] else cv2.INTER_LINEAR
        fg_scaled = cv2.resize(fg_base, (raw_w, raw_h), interpolation=interp)
        mask_scaled = cv2.resize(mask_base, (raw_w, raw_h), interpolation=cv2.INTER_LINEAR)
        fg_warped, mask_warped = random_warp(fg_scaled, mask_scaled)

        if float(mask_warped.sum()) < 8:
            continue

        anchor_label = int(parsing[center_y, center_x])
        center_x = maybe_shift_from_pole(anchor_label, center_x, fg_warped.shape[1], image_w)

        x_tl = int(round(center_x - fg_warped.shape[1] * 0.5))
        y_tl = int(round(center_y - fg_warped.shape[0] * 0.5))
        if x_tl < 0 or y_tl < 0 or x_tl + fg_warped.shape[1] > image_w or y_tl + fg_warped.shape[0] > image_h:
            continue

        bg_patch = base_bg[y_tl:y_tl + fg_warped.shape[0], x_tl:x_tl + fg_warped.shape[1]]
        parsing_patch = parsing[y_tl:y_tl + fg_warped.shape[0], x_tl:x_tl + fg_warped.shape[1]]
        if bg_patch.shape[:2] != fg_warped.shape[:2]:
            continue
        if not patch_is_plausible(parsing_patch, mask_warped):
            continue

        fg_adjusted = mean_std_match(fg_warped, bg_patch, mask_warped)
        bbox = bbox_from_mask(mask_warped, x_tl, y_tl)
        if bbox is None or not is_valid_bbox(bbox):
            continue
        if any(boxes_overlap(bbox, placed_box) for placed_box in occupied_boxes):
            continue

        dst_patch = composite[y_tl:y_tl + fg_warped.shape[0], x_tl:x_tl + fg_warped.shape[1]]
        composite[y_tl:y_tl + fg_warped.shape[0], x_tl:x_tl + fg_warped.shape[1]] = blend_patch(
            dst_patch,
            fg_adjusted,
            mask_warped,
        )
        composite = apply_partial_occlusion(composite, base_bg, bbox)
        update_track_state(track_state, center_x, center_y, scale, image_w, image_h)

        return {
            "bbox": [int(v) for v in bbox],
            "category": track_state["asset"].get("category", "unknown"),
            "source": track_state["asset"].get("source"),
            "track_id": track_state.get("track_id"),
        }

    return None


def apply_motion_blur(img):
    kernel_size = random.choice(MOTION_BLUR_KERNEL_SIZES)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D(
        (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5),
        random.uniform(-MOTION_BLUR_MAX_ANGLE, MOTION_BLUR_MAX_ANGLE),
        1.0,
    )
    kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
    kernel_sum = float(kernel.sum())
    if kernel_sum > 0:
        kernel /= kernel_sum
    return cv2.filter2D(img, -1, kernel)


def apply_sensor_noise(img):
    sigma = random.uniform(2.0, 8.0)
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 255)


def apply_backlight(img):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    center_x = random.uniform(w * 0.3, w * 0.7)
    center_y = random.uniform(h * 0.12, h * 0.38)
    dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    radius = max(h, w) * random.uniform(0.28, 0.42)
    flare = np.exp(-(dist ** 2) / (2.0 * radius * radius)).astype(np.float32)[:, :, None]
    flare_color = np.array([70.0, 90.0, 120.0], dtype=np.float32)
    out = img * 0.86 + flare * flare_color
    return np.clip(out, 0, 255)


def apply_night_mode(img):
    out = img * random.uniform(0.38, 0.62)
    gray = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    cool_tint = np.stack([gray * 0.9, gray * 0.95, gray * 1.12], axis=-1)
    out = out * 0.72 + cool_tint * 0.28
    return np.clip(out, 0, 255)


def apply_fog(img):
    h, _ = img.shape[:2]
    haze_strength = random.uniform(0.08, 0.20)
    vertical = np.linspace(0.3, 1.0, h, dtype=np.float32)[:, None, None]
    fog = 255.0 * vertical * haze_strength
    return np.clip(img * (1.0 - vertical * haze_strength) + fog, 0, 255)


def apply_rain(img):
    h, w = img.shape[:2]
    overlay = np.zeros_like(img, dtype=np.float32)
    streaks = max(40, (h * w) // 18000)
    for _ in range(streaks):
        x0 = random.randint(0, max(w - 1, 0))
        y0 = random.randint(0, max(h - 1, 0))
        length = random.randint(8, 20)
        drift = random.randint(-4, 4)
        x1 = int(np.clip(x0 + drift, 0, max(w - 1, 0)))
        y1 = int(np.clip(y0 + length, 0, max(h - 1, 0)))
        cv2.line(overlay, (x0, y0), (x1, y1), (190, 190, 190), 1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=0.8, sigmaY=0.8)
    return np.clip(img + overlay * random.uniform(0.12, 0.22), 0, 255)


def apply_jpeg_artifacts(img):
    quality = random.randint(45, 92)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, encoded = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8), encode_params)
    if not ok:
        return np.clip(img, 0, 255)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        return np.clip(img, 0, 255)
    return decoded.astype(np.float32)


def apply_dashcam_artifacts(img):
    out = img.astype(np.float32)

    if random.random() < 0.20:
        out = apply_fog(out)
    if random.random() < 0.16:
        out = apply_rain(out)
    if random.random() < MOTION_BLUR_PROB:
        out = apply_motion_blur(out)
    if random.random() < 0.65:
        out = apply_sensor_noise(out)

    out = apply_jpeg_artifacts(out)
    return np.clip(out, 0, 255).astype(np.uint8)


def synthesize(bg, track_states, scene_context):
    """Generate a realistic composite and update per-track temporal state."""
    composite = bg.copy()
    base_bg = bg.copy()
    placements = []
    occupied_boxes = []
    updated_tracks = []

    for track_state in track_states:
        current_state = dict(track_state)
        placement = place_track(composite, base_bg, current_state, scene_context, occupied_boxes)
        if placement is None:
            decay_track_state(current_state)
            updated_tracks.append(current_state)
            continue

        placements.append(placement)
        occupied_boxes.append(tuple(placement["bbox"]))
        updated_tracks.append(current_state)

    composite = apply_dashcam_artifacts(composite)
    return composite, placements, updated_tracks
