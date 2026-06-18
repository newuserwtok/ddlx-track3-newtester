import os
import json
import cv2
import numpy as np


def mask_to_bbox(mask):

    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    bboxes = []

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        bboxes.append([x1, y1, x2, y2])

    return bboxes


def map_bbox_to_1000(bbox, W, H):

    x1, y1, x2, y2 = bbox

    x1 = round(x1 / W * 1000)
    y1 = round(y1 / H * 1000)

    x2 = round(x2 / W * 1000)
    y2 = round(y2 / H * 1000)

    return [x1, y1, x2, y2]


def save_prediction_json(
    save_path,
    pred_mask,
    image_shape,
    threshold=0.5
):

    H, W = image_shape

    pred_mask = (pred_mask > threshold).astype(np.uint8)

    if pred_mask.sum() == 0:

        result = {
            "Bounding boxes": [],
            "Visible forgery traces": "",
            "classification result": "real"
        }

    else:

        bboxes = mask_to_bbox(pred_mask)

        mapped_boxes = []

        for box in bboxes:
            mapped_boxes.append(
                map_bbox_to_1000(box, W, H)
            )

        result = {
            "Bounding boxes": mapped_boxes,
            "Visible forgery traces": "",
            "classification result": "fake"
        }

    with open(save_path, "w") as f:
        json.dump(result, f, indent=4)