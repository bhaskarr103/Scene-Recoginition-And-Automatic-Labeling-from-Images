# src/viz.py
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
import hashlib

def _textsize(draw, text, font):
    """
    Compatibility wrapper for Pillow versions.
    Returns (width, height).
    """
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    else:
        return draw.textsize(text, font=font)

def _class_color(name):
    """
    Deterministic RGBA color for a class name.
    """
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    # brighten slightly
    r = int((r + 160) / 2) if r < 120 else r
    g = int((g + 160) / 2) if g < 120 else g
    b = int((b + 160) / 2) if b < 120 else b
    return (r, g, b, 200)

def draw_annotations(
    pil_img,
    detections,
    keypoint_img=None,
    flow_img=None,
    caption=None,
    object_captions=None,
    object_caption=None,
    show_mask=True,
    mask_alpha=80,
    bbox_width=2,
    caption_above_if_no_space=True
):
    """
    Draw annotations and return an RGB PIL image.

    Handles:
     - per-object bbox + label
     - per-object short caption (below or above bbox if space)
     - mask overlay (resized to image)
     - composite keypoints overlay
     - optional flow inset
     - global caption strip
    """
    # alias support
    if object_caption is not None and object_captions is None:
        object_captions = object_caption

    img = pil_img.convert("RGBA")
    W, H = img.size

    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)

    # fonts with fallback
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 12)
        font_caption = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_caption = ImageFont.load_default()

    for idx, d in enumerate(detections):
        try:
            x0, y0, x1, y1 = map(int, d['bbox'])
        except Exception:
            continue

        cls = str(d.get('class', "obj"))
        score = d.get('score', None)
        label = f"{cls}" + (f" {score:.2f}" if score is not None else "")

        color = _class_color(cls)
        color_rgb = (color[0], color[1], color[2])

        # draw bbox (multiple-offset for width)
        for off in range(bbox_width):
            draw.rectangle([x0-off, y0-off, x1+off, y1+off], outline=color_rgb + (255,), width=1)

        # label header
        tw, th = _textsize(draw, label, font)
        lab_x0 = x0
        lab_y0 = max(0, y0 - th - 6)
        lab_x1 = min(W, x0 + tw + 8)
        lab_y1 = y0
        draw.rectangle([lab_x0, lab_y0, lab_x1, lab_y1], fill=(0,0,0,200))
        draw.text((lab_x0 + 3, lab_y0 + 2), label, fill=(255,255,255,255), font=font)

        # mask overlay
        if show_mask and d.get('mask') is not None:
            try:
                mask_arr = d['mask']
                mask_img = Image.fromarray(mask_arr).convert("L")
                if mask_img.size != img.size:
                    mask_img = mask_img.resize(img.size, resample=Image.NEAREST)
                color_layer = Image.new("RGBA", img.size, (color[0], color[1], color[2], mask_alpha))
                img.paste(color_layer, (0,0), mask_img)
            except Exception:
                pass

        # per-object caption block
        if object_captions and idx in object_captions:
            obj_text = str(object_captions[idx])
            wrapped = textwrap.fill(obj_text, width=28)
            lines = wrapped.split("\n")
            line_sizes = [_textsize(draw, line, font_small) for line in lines]
            line_heights = [h for (w,h) in line_sizes]
            line_widths = [w for (w,h) in line_sizes]
            total_h = sum(line_heights) + (len(lines)-1)*2
            max_w = max(line_widths) if line_widths else 0

            # try below bbox
            rect_x0 = x0
            rect_y0 = y1 + 6
            rect_x1 = min(W, x0 + max_w + 12)
            rect_y1 = rect_y0 + total_h + 8

            # if not enough room below, optionally place above bbox
            if rect_y1 > H:
                if caption_above_if_no_space:
                    rect_y1_alt = y0 - 6
                    rect_y0_alt = rect_y1_alt - (total_h + 8)
                    # if alt yields negative, clamp to top
                    if rect_y0_alt < 0:
                        rect_y0_alt = 0
                        rect_y1_alt = min(H, rect_y0_alt + total_h + 8)
                    rect_x0 = x0
                    rect_y0 = rect_y0_alt
                    rect_x1 = min(W, x0 + max_w + 12)
                    rect_y1 = rect_y1_alt
                else:
                    # clamp to fit inside image bottom
                    rect_y1 = H
                    rect_y0 = max(0, H - (total_h + 8))

            # ensure rect coords valid
            if rect_y1 <= rect_y0:
                rect_y1 = rect_y0 + max(1, total_h + 4)
                if rect_y1 > H:
                    rect_y1 = H
                    rect_y0 = max(0, rect_y1 - (total_h + 4))

            draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=(0,0,0,180))
            y_text = rect_y0 + 4
            for i, line in enumerate(lines):
                draw.text((rect_x0 + 6, y_text), line, fill=(255,255,255,255), font=font_small)
                y_text += line_heights[i] + 2

    # blend keypoints overlay if given
    if keypoint_img is not None:
        try:
            key = keypoint_img.convert("RGBA").resize(img.size)
            img = Image.alpha_composite(img, key)
        except Exception:
            pass

    # composite overlay (bboxes/labels/captions)
    img = Image.alpha_composite(img, overlay)

    # flow inset
    if flow_img is not None:
        try:
            inset_w = max(120, W // 4)
            inset_h = int(flow_img.height * (inset_w / flow_img.width))
            inset = flow_img.convert("RGB").resize((inset_w, inset_h))
            inset_pos = (W - inset_w - 8, H - inset_h - 8)
            img.paste(inset, inset_pos)
        except Exception:
            pass

    # global caption strip
    if caption:
        draw2 = ImageDraw.Draw(img)
        wrap_caption = textwrap.fill(caption, width=90)
        lines = wrap_caption.split("\n")
        line_h = _textsize(draw2, "A", font=font_caption)[1]
        strip_h = max(60, len(lines) * (line_h + 4) + 12)
        draw2.rectangle([0, H - strip_h, W, H], fill=(0,0,0,200))
        draw2.multiline_text((8, H - strip_h + 8), wrap_caption, fill=(255,255,255,255), font=font_caption, spacing=4)

    return img.convert("RGB")
