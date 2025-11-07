# eval_metrics.py
"""
Evaluation harness for your Scene Recognition & Automatic Labeling project.
Plug-and-play with the code in src/.

Produces:
 - eval_metrics.csv with per-image rows
 - prints a summary with mean times and CLIP scores

Usage:
  python eval_metrics.py
Options (edit at top of file):
  DATA_DIR, OUTPUT_DIR, USE_FLOW, TOP_K_CAPTIONS, GT_INTERACTIONS_JSON
"""

import os
import time
from pathlib import Path
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP for scoring
from transformers import CLIPProcessor, CLIPModel

# import your modules (these match the files you uploaded)
from src.detect import detect as detect_fn
from src.actions import infer_actions
from src.caption import generate_caption, generate_object_captions
from src.viz import draw_annotations
from src.flow import dense_optical_flow  # optional, used only if pairs provided

# ---------------- CONFIG ----------------
DATA_DIR = Path("data")           # images (.jpg) to score
OUTPUT_DIR = Path("output")
OUT_CSV = Path("eval_metrics.csv")
USE_FLOW = False                # set True if you have pairs and want flow
TOP_K_CAPTIONS = 8
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
GT_INTERACTIONS_JSON = None     # optional: path to GT interaction json lines (see format below)
# ---------------------------------------

# prepare CLIP
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)

def clip_score_image_text(pil_image, text):
    """Return CLIP image-text cosine similarity (float)"""
    try:
        inputs = clip_processor(text=[text], images=pil_image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = clip_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        score = (image_features * text_features).sum(dim=-1).cpu().item()
        return float(score)
    except Exception as e:
        print("CLIP scoring failed:", e)
        return float("nan")

def score_images(image_paths):
    rows = []
    times = {"detect":[], "flow":[], "cap_global":[], "cap_objs":[], "viz":[], "total":[]}
    for img_path in tqdm(image_paths, desc="Evaluating"):
        img = Image.open(img_path).convert("RGB")
        name = Path(img_path).name
        t0 = time.time()

        # detection
        t = time.time()
        detections = detect_fn(img)  # returns list of dicts
        times["detect"].append(time.time()-t)

        # optional flow - here we skip since we usually don't have pairs
        flow = None
        if USE_FLOW:
            # For flow you must provide a pair; this template assumes second image naming convention or list.
            # If you want flow, adapt this to use paired frames.
            t = time.time()
            # flow, flow_vis = dense_optical_flow(img, second_img)
            times["flow"].append(time.time()-t)
        else:
            times["flow"].append(0.0)

        # produce facts for captioning + action inference
        # infer_actions expects detections list and optional flow -> returns facts [(class,score,bbox,mask,attrs,action), ...]
        facts = infer_actions(detections, flow=flow)

        # global caption (BLIP)
        t = time.time()
        global_cap = generate_caption(img, facts=facts)
        times["cap_global"].append(time.time()-t)

        # CLIP global score
        clip_global = clip_score_image_text(img, global_cap)

        # object-level captions (uses facts)
        t = time.time()
        obj_caps_dict = generate_object_captions(img, facts, top_k=TOP_K_CAPTIONS)
        # obj_caps_dict is mapping detection_index->short caption
        times["cap_objs"].append(time.time()-t)

        # CLIP object-level mean score
        obj_scores = []
        # try to compute CLIP score per crop; need to re-create crops
        for idx, cap_text in obj_caps_dict.items():
            try:
                # find corresponding bbox from facts (facts preserves order)
                if idx < len(facts):
                    _, score, bbox, mask, attrs, action = facts[idx]
                    x0,y0,x1,y1 = map(int, bbox)
                    crop = img.crop((x0,y0,x1,y1)).convert("RGB")
                    sc = clip_score_image_text(crop, cap_text)
                    obj_scores.append(sc)
            except Exception:
                pass
        clip_objs_mean = float(np.nanmean(obj_scores)) if obj_scores else float("nan")

        # visualization
        t = time.time()
        # draw_annotations signature: (pil_img, detections, keypoint_img=None, flow_img=None, caption=None, object_captions=None, ...)
        annotated = draw_annotations(img, detections, caption=global_cap, object_captions=obj_caps_dict)
        outfile = OUTPUT_DIR / f"annotated_{name}"
        OUTPUT_DIR.mkdir(exist_ok=True)
        annotated.save(outfile)
        times["viz"].append(time.time()-t)

        t_total = time.time() - t0
        times["total"].append(t_total)

        # build row
        row = {
            "image": name,
            "n_detections": len(detections),
            "global_caption": global_cap,
            "clip_global": clip_global,
            "clip_objects_mean": clip_objs_mean,
            "time_detect": times["detect"][-1],
            "time_flow": times["flow"][-1],
            "time_cap_global": times["cap_global"][-1],
            "time_cap_objects": times["cap_objs"][-1],
            "time_viz": times["viz"][-1],
            "time_total": t_total,
            "annotated_image": str(outfile)
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = {
        "mean_time_total": float(df["time_total"].mean()),
        "median_time_total": float(df["time_total"].median()),
        "mean_clip_global": float(df["clip_global"].mean(skipna=True)),
        "mean_clip_objects": float(df["clip_objects_mean"].mean(skipna=True)),
        "mean_detections": float(df["n_detections"].mean())
    }
    return df, summary

# optional: evaluate interactions if GT provided
def evaluate_interactions_from_files(predicted_facts_dict, gt_json_path):
    """
    predicted_facts_dict: dict image_name -> list of tuples (person_idx, object_idx, action_label) OR facts extracted
    GT format (JSON lines) (optional):
      {"image":"img1.jpg", "pairs":[ {"person_bbox":[x0,y0,x1,y1], "object_bbox":[...], "action":"riding"} , ... ] }
    This function is a helper but GT evaluation requires you to supply GT in expected format.
    """
    import json
    with open(gt_json_path, "r") as f:
        gt_lines = [json.loads(l) for l in f if l.strip()]
    # Implement matching based on IoU and same action label. This is left as a small extension if you have GT.
    print("GT evaluation helper loaded - implement matching logic if you have GT.")
    return None

if __name__ == "__main__":
    # collect images
    img_files = sorted([str(p) for p in Path(DATA_DIR).glob("*.jpg")])
    if len(img_files) == 0:
        print("No images found in", DATA_DIR, "— please add .jpg files or edit DATA_DIR.")
        raise SystemExit(1)

    print("Device for models:", device)
    print("Using CLIP model:", CLIP_MODEL_NAME)
    df, summ = score_images(img_files)
    df.to_csv(OUT_CSV, index=False)
    print("\nSaved metrics to", OUT_CSV)
    print("Summary:")
    for k,v in summ.items():
        print(f"  {k}: {v}")
