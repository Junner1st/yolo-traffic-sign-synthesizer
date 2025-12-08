from ultralytics import YOLO
import cv2
import numpy as np
import torch
from model.BiSeNet.lib.models.bisenetv2 import BiSeNetV2
import os
import subprocess


pal = np.array([
    [128, 64,128], [244,35,232], [70,70,70],  [102,102,156],
    [190,153,153], [153,153,153],[250,170,30],[220,220, 0],
    [107,142, 35], [152,251,152],[70,130,180],[220,20,60],
    [255, 0, 0],   [0,0,142],    [0,60,100],   [0,80,100],
    [0,0,230],     [119,11,32]
])

def load_bisenet_model():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _model_path = os.path.join(_script_dir, "model_final_v2_city.pth")
    _url = "https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/model_final_v2_city.pth"

    if not os.path.exists(_model_path):
        print(f"Model not found at {_model_path}, downloading...")
        try:
            subprocess.run(["wget", _url, "-O", _model_path], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("Failed to download model_final_v2_city.pth. Ensure wget is installed and network is available.") from exc

    net = BiSeNetV2(n_classes=19)
    net.load_state_dict(torch.load(_model_path, map_location='cpu'))
    net.eval()
    return net


def segment_with_bisenet(img, net):
    """Run BiSeNet and return parsing mask.
    img: input BGR image
    net: BiSeNet model
    """
    h0, w0 = img.shape[:2]
    img_resized = cv2.resize(img, (1024, 512))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    inp = img_rgb.transpose(2, 0, 1)
    inp = torch.from_numpy(inp).float() / 255.
    inp = inp.unsqueeze(0)

    with torch.no_grad():
        out = net(inp)[0]
        parsing = out.argmax(1).squeeze().cpu().numpy()

    parsing = cv2.resize(parsing, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return parsing


def mask_to_color(parsing, palette=pal):
    """Convert (H, W) parsing mask to color map.
    parsing: class mask (H, W)
    palette: color lookup table (num_classes, 3)
    """
    palette = np.asarray(palette)
    max_label = int(np.max(parsing))
    if max_label >= palette.shape[0]:
        pad_rows = max_label - palette.shape[0] + 1
        pad = np.tile(np.array([[0, 0, 0]], dtype=palette.dtype), (pad_rows, 1))
        palette = np.vstack([palette, pad])

    color_map = palette[parsing]
    return color_map.astype(np.uint8)

if __name__=="__main__":
    # YOLOv8 segmentation
    model = YOLO("yolov8n-seg.pt")

    img_path = "../data/not_synthesized/McL8WnjMvxM_60s_frame_18.jpg"
    img = cv2.imread(img_path)

    # Method 1: YOLOv8
    results = model(img)[0]
    mask_vis_yolo = img.copy()
    if results.masks is not None:
        for m in results.masks.data:
            m = m.cpu().numpy()
            m = (m * 255).astype(np.uint8)
            m = cv2.resize(m, (img.shape[1], img.shape[0]))
            mask_color = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)
            colored = np.zeros_like(img)
            colored[:, :] = mask_color
            mask_vis_yolo = np.where(m[..., None] > 0, colored, mask_vis_yolo)

    cv2.imwrite("yolo_seg_vis.png", mask_vis_yolo)

    # Method 2: BiSeNet
    bisenet_net = load_bisenet_model()
    parsing = segment_with_bisenet(img, bisenet_net)


    mask_color_map_vis_bisenet = mask_to_color(parsing, pal)

    cv2.imwrite("bisenet_seg_vis.png", mask_color_map_vis_bisenet)