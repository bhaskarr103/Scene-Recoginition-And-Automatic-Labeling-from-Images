# src/caption.py
"""
Image-grounded captioning using BLIP (Salesforce/blip-image-captioning-base).
Provides:
 - generate_caption(image_pil, facts, ...)
 - generate_object_captions(image_pil, facts, top_k=8, ...)
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import textwrap
from PIL import Image

# --- Configuration ---
_MODEL_NAME = "Salesforce/blip-image-captioning-base"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # will be 'cpu' on your setup

# --- Lazy init (model download happens on first import/use) ---
_processor = None
_model = None

def _init_model():
    global _processor, _model
    if _processor is None or _model is None:
        print(f"Loading BLIP model ({_MODEL_NAME}) to device {_DEVICE} — this may download weights on first run.")
        _processor = BlipProcessor.from_pretrained(_MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(_MODEL_NAME)
        _model.to(_DEVICE)
    return _processor, _model

# --- Helpers ---
def _fallback_caption(facts):
    """Small deterministic fallback if model fails."""
    if not facts:
        return "An image."
    top = sorted(facts, key=lambda x: x[1], reverse=True)[:3]
    labels = [f[0] for f in top]
    return "A scene with " + ", ".join(labels) + "."

def _clean_text(s, width=220):
    s = s.strip()
    # remove leading punctuation/spaces
    while len(s) > 0 and s[0] in " \n\r\t-:;,.()[]\"'":
        s = s[1:]
    return textwrap.shorten(s.split("\n")[0].strip(), width=width, placeholder="...")

# --- Public functions ---

def generate_caption(image_pil, facts=None, max_new_tokens=50, num_beams=3, do_sample=False, verbose=False):
    """
    Generate a global caption for the full image using BLIP.
    Args:
      image_pil: PIL.Image (RGB) of the full image
      facts: optional structured facts (not used to condition BLIP here, kept for compatibility)
    Returns:
      caption string
    """
    try:
        processor, model = _init_model()
        # BLIP expects RGB PIL images or tensors
        inputs = processor(images=image_pil, return_tensors="pt").to(_DEVICE)
        # generate
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=do_sample)
        caption = processor.decode(gen[0], skip_special_tokens=True)
        caption = _clean_text(caption)
        if verbose:
            print("BLIP global caption:", caption)
        if not caption:
            return _fallback_caption(facts or [])
        return caption
    except Exception as e:
        if verbose:
            print("BLIP generate_caption failed:", e)
        return _fallback_caption(facts or [])

def generate_object_captions(image_pil, facts, top_k=8, max_new_tokens=40, num_beams=3, do_sample=False, min_box_area=20, verbose=False):
    """
    Generate per-object captions by cropping the detection bounding boxes and running BLIP on each crop.
    Args:
      image_pil: PIL.Image (RGB)
      facts: list of tuples (class, score, bbox, mask, attrs, action)
      top_k: only generate captions for top_k most confident detections
    Returns:
      dict mapping detection index -> short caption string
    """
    if not facts:
        return {}
    try:
        processor, model = _init_model()
        # sort by score descending with original indices
        indexed = list(enumerate(facts))
        indexed_sorted = sorted(indexed, key=lambda x: x[1][1], reverse=True)
        selected = indexed_sorted[:top_k]
        obj_caps = {}
        W, H = image_pil.size

        for idx, (cls, score, bbox, mask, attrs, action) in selected:
            x0, y0, x1, y1 = map(int, bbox)
            # clamp
            x0 = max(0, min(W-1, x0)); x1 = max(0, min(W, x1))
            y0 = max(0, min(H-1, y0)); y1 = max(0, min(H, y1))
            if x1 <= x0 or y1 <= y0:
                if verbose:
                    print(f"Skipping invalid bbox for idx {idx}: {bbox}")
                continue
            area = (x1-x0)*(y1-y0)
            if area < min_box_area:
                if verbose:
                    print(f"Skipping small bbox for idx {idx}: area={area}")
                continue

            try:
                crop = image_pil.crop((x0, y0, x1, y1)).convert("RGB")
                inputs = processor(images=crop, return_tensors="pt").to(_DEVICE)
                gen = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=do_sample)
                text = processor.decode(gen[0], skip_special_tokens=True)
                short = _clean_text(text, width=80)
                if not short:
                    short = f"{cls} ({action})"
                obj_caps[idx] = short
                if verbose:
                    print(f"Obj {idx} [{cls}] -> {short}")
            except Exception as e:
                if verbose:
                    print(f"Failed BLIP for object {idx}: {e}")
                obj_caps[idx] = f"{cls} ({action})"
        return obj_caps
    except Exception as e:
        if verbose:
            print("generate_object_captions failed:", e)
        # fallback: simple labels
        return {i: f"{f[0]} ({f[5]})" for i, f in enumerate(facts[:top_k])}
