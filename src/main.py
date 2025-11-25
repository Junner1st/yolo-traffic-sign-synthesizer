import cv2
import numpy as np
import random

from resize_and_horizon_perspective_enhancement import detect_horizon
from semantic_segmentation_bisenet_yolov8seg import load_bisenet_model, segment_with_bisenet, mask_to_color
import cv2
import numpy as np
import random



def compute_perspective_scale(y, horizon, k=800, min_scale=0.05, max_scale=0.4):
    """Compute scale factor based on distance from horizon.
    y: y-position
    horizon: horizon row
    k: scaling constant
    """
    dist = abs(y - horizon)
    scale = k / max(dist, 1)
    return np.clip(scale, min_scale, max_scale)


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
    """Create feathered alpha mask.
    fg: foreground image
    feather: soft edge width
    """
    h, w = fg.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)

    mask[:feather,:] *= np.linspace(0,1,feather)[:,None]
    mask[-feather:,:] *= np.linspace(1,0,feather)[:,None]
    mask[:, :feather] *= np.linspace(0,1,feather)[None,:]
    mask[:, -feather:] *= np.linspace(1,0,feather)[None,:]

    mask = cv2.GaussianBlur(mask, (feather*2+1, feather*2+1), 0)
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


def synthesize(bg, fg_list, bisenet_model, n_objects=3):
    """Generate synthetic composite.
    bg: background BGR
    fg_list: list of traffic sign images
    bisenet_model: BiSeNet network
    n_objects: number of signs to place
    """
    h, w = bg.shape[:2]
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    horizon = detect_horizon(gray)

    parsing = segment_with_bisenet(bg, bisenet_model)

    
    cv2.imwrite("main_bisenet_seg_vis.png", mask_to_color(parsing))


    SKY = 23
    ROAD = 7

    ys = np.random.randint(horizon+20, h-20, size=n_objects)
    xs = np.random.randint(20, w-20, size=n_objects)

    for (x, y) in zip(xs, ys):
        if parsing[y, x] in [SKY, ROAD]:
            continue

        fg_raw = random.choice(fg_list)
        scale = compute_perspective_scale(y, horizon)
        new_w = int(fg_raw.shape[1] * scale)
        new_h = int(fg_raw.shape[0] * scale)
        fg_scaled = cv2.resize(fg_raw, (new_w, new_h), interpolation=cv2.INTER_AREA)

        y1 = max(0, y-new_h//2)
        y2 = min(h, y+new_h//2)
        x1 = max(0, x-new_w//2)
        x2 = min(w, x+new_w//2)
        bg_patch = bg[y1:y2, x1:x2]

        fg_adj = mean_std_match(fg_scaled, bg_patch)
        fg_warped = random_warp(fg_adj)

        bg = paste_object(bg, fg_warped, x, y)

    return bg


if __name__ == "__main__":
    # bg = cv2.imread("../data/not_synthesized/McL8WnjMvxM_58s_frame_24.jpg")
    bg = cv2.imread("resized_bg.jpg")
    fg1 = cv2.imread("../data/temp/train_plus_禁止左轉_19.png")
    fg2 = cv2.imread("../data/temp/train_plus_40_61.png")

    # Example: load bisenet
    bisenet = load_bisenet_model()

    output = synthesize(bg, [fg1, fg2], bisenet, n_objects=3)
    cv2.imwrite("synthetic_output.jpg", output)
