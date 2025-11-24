import cv2
import numpy as np


def detect_horizon(gray, smooth=31):
    """
    Use vertical gradient to find the row with maximum brightness change ≈ horizon.
    """
    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.GaussianBlur(sobel, (smooth, smooth), 0)

    row_scores = np.mean(np.abs(sobel), axis=1)
    horizon_row = np.argmax(row_scores)
    return horizon_row


def resize_preserve_horizon(img, scale=0.7, contrast_boost=1.25):
    """
    Scale proportionally + maintain horizon proportional position + enhance contrast near vanishing point.
    """
    h, w = img.shape[:2]

    new_h = int(h * scale)
    new_w = int(w * scale)

    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Re-estimate horizon
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    horizon = detect_horizon(gray)

    # Create a mask where boost increases closer to horizon
    mask = np.zeros_like(gray, dtype=np.float32)
    for y in range(new_h):
        d = abs(y - horizon)
        boost = np.exp(-(d / (0.15 * new_h))**2)   # Gaussian boost
        mask[y, :] = boost

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=15)
    mask = mask[..., None]

    # Enhance contrast in that region
    small_f = small.astype(np.float32)
    enhanced = small_f * (1.0 + (contrast_boost - 1.0) * mask)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced, horizon


if __name__ == "__main__":
    img = cv2.imread("../data/not_synthesized/McL8WnjMvxM_60s_frame_18.jpg")

    out, hz = resize_preserve_horizon(img, scale=0.7)

    cv2.imwrite("resized_horizon.jpg", out)
