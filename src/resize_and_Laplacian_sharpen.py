import cv2
import numpy as np


def resize_with_detail_enhancement(img, scale=0.7, alpha=0.3):
    h, w = img.shape[:2]

    new_h = int(h * scale)
    new_w = int(w * scale)

    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # energy (float32)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lap_energy = cv2.Laplacian(gray, cv2.CV_32F)
    energy = np.abs(lap_energy)

    # detail enhancement (uint8)
    lap_detail = cv2.Laplacian(small, cv2.CV_64F)
    lap_detail = cv2.convertScaleAbs(lap_detail)
    
    enhanced = cv2.addWeighted(small, 1.0, lap_detail, alpha, 0)

    return enhanced, energy, lap_detail


if __name__ == "__main__":
    img = cv2.imread("../data/not_synthesized/McL8WnjMvxM_60s_frame_18.jpg")

    out, energy, lap_detail = resize_with_detail_enhancement(img, scale=0.7, alpha=0.3)

    heat_uint8 = np.uint8(255 * (energy - energy.min()) / (energy.max() - energy.min() + 1e-8))
    heatmap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(out, 0.6, heatmap, 0.4, 0)

    cv2.imwrite("energy_heatmap.png", heatmap)
    cv2.imwrite("energy_overlay.png", overlay)
    cv2.imwrite("lap_detail_raw.png", lap_detail)
    cv2.imwrite("resized_bg.jpg", out)
