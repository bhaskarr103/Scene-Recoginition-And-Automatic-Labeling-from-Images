# src/features.py
import cv2
import numpy as np
from PIL import Image

def compute_keypoints_and_draw(pil_image, method='SIFT', max_kp=300):
    """
    Compute classical keypoints and return an image with keypoints drawn and keypoint list.
    Returns: (PIL image with keypoints drawn, list of cv2.KeyPoint)
    """
    img = np.array(pil_image.convert('RGB'))[:, :, ::-1]  # RGB->BGR for OpenCV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    method = method.upper()
    if method == 'SIFT':
        try:
            detector = cv2.SIFT_create()
        except AttributeError:
            detector = cv2.xfeatures2d.SIFT_create()
    else:
        detector = cv2.ORB_create(nfeatures=max_kp)
    kps = detector.detect(gray, None)
    # Sort by response and limit
    kps = sorted(kps, key=lambda k: -k.response)[:max_kp]
    out = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out = out[:, :, ::-1]  # BGR->RGB
    return Image.fromarray(out), kps
