# src/flow.py
import cv2
import numpy as np
from PIL import Image

def dense_optical_flow(pil_img1, pil_img2):
    """
    Compute Farneback dense optical flow between two PIL images.
    Returns (flow_array (H,W,2), flow_visualization PIL image RGB)
    """
    img1 = cv2.cvtColor(np.array(pil_img1.convert('RGB')), cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(np.array(pil_img2.convert('RGB')), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    h, w = img1.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[...,2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return flow, Image.fromarray(rgb)

def flow_region_magnitude(flow, bbox):
    """
    Mean magnitude inside bbox [x0,y0,x1,y1]
    """
    if flow is None:
        return 0.0
    x0,y0,x1,y1 = map(int, bbox)
    h,w,_ = flow.shape
    x0 = max(0, min(w-1, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h-1, y0)); y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub = flow[y0:y1, x0:x1]
    if sub.size == 0:
        return 0.0
    mag = np.sqrt(sub[...,0]**2 + sub[...,1]**2)
    return float(mag.mean())
