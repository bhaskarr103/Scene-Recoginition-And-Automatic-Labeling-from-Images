# run_demo.py
import os, sys
from PIL import Image
import matplotlib.pyplot as plt

# safe project root detection
try:
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
except NameError:
    PROJECT_ROOT = os.path.abspath(".")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# imports (from src/*.py)
from src.detect import detect
from src.features import compute_keypoints_and_draw
from src.flow import dense_optical_flow
from src.actions import infer_actions
from src.caption import generate_caption, generate_object_captions
from src.viz import draw_annotations

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_PATH = os.path.join(DATA_DIR, "input.jpg")
FRAME0 = os.path.join(DATA_DIR, "frame_000.jpg")
FRAME1 = os.path.join(DATA_DIR, "frame_001.jpg")
OUT_PATH = os.path.join(OUTPUT_DIR, "annotated.jpg")

def main():
    print("=== Scene Prose CV ===")
    if not os.path.exists(IMG_PATH):
        raise FileNotFoundError(f"Place an image at: {IMG_PATH}")

    img = Image.open(IMG_PATH).convert("RGB")

    print("1) Detecting objects...")
    dets = detect(img, score_thresh=0.6, device="cpu")
    print(f"Detected {len(dets)} objects")

    print("2) Extracting keypoints (SIFT)...")
    kp_img, kps = compute_keypoints_and_draw(img, method="SIFT", max_kp=300)
    print(f"Keypoints: {len(kps)}")

    print("3) Optical flow (optional)...")
    flow = None; flow_vis = None
    if os.path.exists(FRAME0) and os.path.exists(FRAME1):
        im0 = Image.open(FRAME0).convert("RGB")
        im1 = Image.open(FRAME1).convert("RGB")
        flow, flow_vis = dense_optical_flow(im0, im1)
        print("Flow computed.")
    else:
        print("Frame pair not found, skipping flow.")

    print("4) Inferring actions...")
    facts = infer_actions(dets, flow=flow, flow_thresh=1.0)

    print("5) Generating captions...")

    global_caption = generate_caption(img, facts, verbose=True)         # verbose True for debug
    print("Global caption:", global_caption)
    object_caps = generate_object_captions(img, facts, top_k=8, verbose=True)
    print("Per-object caps for indices:", list(object_caps.keys()))

    print("6) Drawing and saving annotated image...")
    final = draw_annotations(img, dets, keypoint_img=kp_img, flow_img=flow_vis,
                             caption=global_caption, object_captions=object_caps)
    final.save(OUT_PATH)
    print("Saved:", OUT_PATH)

    try:
        plt.figure(figsize=(10,8)); plt.imshow(final); plt.axis("off"); plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
