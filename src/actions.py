# src/actions.py
from src.flow import flow_region_magnitude

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(1e-6, (boxA[2]-boxA[0])*(boxA[3]-boxA[1]))
    areaB = max(1e-6, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
    return inter / (areaA + areaB - inter)

def infer_actions(detections, flow=None, flow_thresh=1.0):
    """
    detections: list of dicts as returned by detect()
    flow: dense flow array (H,W,2) or None
    Returns list of tuples: (class, score, bbox, mask, attrs_dict, action)
    """
    persons = [d for d in detections if d['class']=='person']
    others = [d for d in detections if d['class']!='person']
    facts = []
    for d in detections:
        action = 'present'
        # motion cue
        if flow is not None:
            mag = flow_region_magnitude(flow, d['bbox'])
            if mag > flow_thresh:
                action = 'moving'
        # person interactions heuristics
        if d['class'] == 'person':
            for o in others:
                if o['class'] in ('bicycle','motorcycle') and iou(d['bbox'], o['bbox']) > 0.12:
                    action = f"riding {o['class']}"
                    break
                if o['class'] in ('dog','cat') and iou(d['bbox'], o['bbox']) > 0.05:
                    action = f"interacting with {o['class']}"
        facts.append((d['class'], d['score'], d['bbox'], d.get('mask'), {}, action))
    return facts
