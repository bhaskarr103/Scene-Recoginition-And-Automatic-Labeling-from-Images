# src/detect.py
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Load Mask R-CNN (pretrained on COCO). CPU by default.
MODEL = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
MODEL.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse',
    'sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
    'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard',
    'surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana',
    'apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]

TRANSFORM = T.Compose([T.ToTensor()])

def detect(image_pil, score_thresh=0.6, device='cpu'):
    """
    Run Mask R-CNN on a PIL image.
    Returns: list of dicts: {class, score, bbox [x0,y0,x1,y1], mask (HxW uint8 0..255 or None)}
    """
    img_t = TRANSFORM(image_pil).to(device)
    with torch.no_grad():
        MODEL.to(device)
        outs = MODEL([img_t])[0]

    detections = []
    scores = outs.get('scores', []).cpu().numpy()
    labels = outs.get('labels', []).cpu().numpy()
    boxes = outs.get('boxes', []).cpu().numpy()
    masks = outs.get('masks', None)

    for i, s in enumerate(scores):
        if s < score_thresh:
            continue
        label = int(labels[i])
        bbox = boxes[i].tolist()
        mask = None
        if masks is not None:
            # mask is returned as float [1,H,W], convert to uint8 0..255
            mask = (masks[i, 0].mul(255).byte().cpu().numpy())
        detections.append({
            'class': COCO_INSTANCE_CATEGORY_NAMES[label] if label < len(COCO_INSTANCE_CATEGORY_NAMES) else str(label),
            'score': float(s),
            'bbox': [float(x) for x in bbox],
            'mask': mask
        })
    return detections
