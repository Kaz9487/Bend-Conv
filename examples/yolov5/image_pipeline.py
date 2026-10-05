"""Shared image preprocessing and class-aware NMS; never used for Bend tensor math."""

from pathlib import Path
import numpy as np


def preprocess(image, size):
    import cv2  # only image decoding and drawing need OpenCV

    im = cv2.imread(str(image))
    if im is None:
        raise FileNotFoundError(image)
    h, w = im.shape[:2]
    ratio = min(size / h, size / w)
    rw, rh = round(w * ratio), round(h * ratio)
    resized = (
        cv2.resize(im, (rw, rh), interpolation=cv2.INTER_LINEAR)
        if (rw, rh) != (w, h)
        else im.copy()
    )
    dw, dh = (size - rw) / 2, (size - rh) / 2
    left, right = round(dw - 0.1), round(dw + 0.1)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    x = np.ascontiguousarray(
        padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32
    ) / np.float32(255)
    return x, dict(ratio=ratio, pad=[dw, dh], original_shape=[h, w], size=size)


def iou_one(box, boxes):
    inter = np.maximum(
        0, np.minimum(box[2:4], boxes[:, 2:4]) - np.maximum(box[:2], boxes[:, :2])
    ).prod(1)
    area = np.maximum(0, box[2:4] - box[:2]).prod()
    areas = np.maximum(0, boxes[:, 2:4] - boxes[:, :2]).prod(1)
    return inter / (area + areas - inter + 1e-7)


def nms(pred, conf=0.25, iou=0.45, max_det=300):
    p = pred[0]
    p = p[p[:, 4] > conf]
    if not len(p):
        return np.empty((0, 6), np.float32)
    scores = p[:, 5:] * p[:, 4:5]
    classes = scores.argmax(1)
    confidence = scores[np.arange(len(p)), classes]
    mask = confidence > conf
    p, classes, confidence = p[mask], classes[mask], confidence[mask]
    boxes = np.concatenate((p[:, :2] - p[:, 2:4] / 2, p[:, :2] + p[:, 2:4] / 2), 1)
    order = np.argsort(-confidence, kind='stable')[:30000]
    keep = []
    while len(order) and len(keep) < max_det:
        j = order[0]
        keep.append(j)
        rest = order[1:]
        order = rest[(classes[rest] != classes[j]) | (iou_one(boxes[j], boxes[rest]) <= iou)]
    return np.concatenate((boxes[keep], confidence[keep, None], classes[keep, None]), 1).astype(
        np.float32
    )


def draw(image, detections, info, names, output):
    import cv2

    im = cv2.imread(str(image))
    for d in detections:
        box = d[:4].copy()
        box[[0, 2]] -= info['pad'][0]
        box[[1, 3]] -= info['pad'][1]
        box /= info['ratio']
        box[[0, 2]] = np.clip(box[[0, 2]], 0, im.shape[1])
        box[[1, 3]] = np.clip(box[[1, 3]], 0, im.shape[0])
        x1, y1, x2, y2 = box.astype(int)
        color = (60, 210, 60)
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        label = f'{names[int(d[5])]} {d[4]:.3f}'
        cv2.putText(im, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.imwrite(str(output), im)
