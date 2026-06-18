import os
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm

import albumentations as albu
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, DataLoader

from IMDLBenCo.registry import MODELS

import IMDLBenCo.training_scripts.utils.misc as misc
from model.MaskCLIP import MaskCLIP


# =========================
# Dataset
# =========================

class CompetitionDataset(Dataset):

    def __init__(self, image_dir, image_size=512):

        self.image_dir = image_dir

        self.files = sorted(os.listdir(image_dir))

        self.transform = albu.Compose([
            albu.Resize(image_size, image_size),
            albu.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        fname = self.files[idx]

        path = os.path.join(self.image_dir, fname)

        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W = image.shape[:2]

        transformed = self.transform(image=image)

        image = transformed["image"]

        return {
            "image": image,
            "name": fname,
            "shape": (H, W)
        }


# =========================
# bbox
# =========================

def mask_to_bbox(mask):

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    bboxes = []

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)

        bboxes.append([
            x,
            y,
            x + w,
            y + h
        ])

    return bboxes


def map_box(box, W, H):

    x1, y1, x2, y2 = box

    x1 = round(x1 / W * 1000)
    y1 = round(y1 / H * 1000)

    x2 = round(x2 / W * 1000)
    y2 = round(y2 / H * 1000)

    return [x1, y1, x2, y2]


# =========================
# json
# =========================

def save_json(save_path, mask, shape, threshold=0.5):

    H, W = shape

    mask = (mask > threshold).astype(np.uint8)

    if mask.sum() == 0:

        result = {
            "Bounding boxes": [],
            "Visible forgery traces": "",
            "classification result": "real"
        }

    else:

        boxes = mask_to_bbox(mask)

        mapped_boxes = []

        for box in boxes:
            mapped_boxes.append(
                map_box(box, W, H)
            )

        result = {
            "Bounding boxes": mapped_boxes,
            "Visible forgery traces": "",
            "classification result": "fake"
        }

    with open(save_path, "w") as f:
        json.dump(result, f, indent=4)


# =========================
# inference
# =========================

@torch.no_grad()
def inference(model, loader, device, save_dir):

    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    for batch in tqdm(loader):

        images = batch["image"].to(device)

        outputs = model(images)

        pred_masks = outputs["pred_mask"]

        pred_masks = pred_masks.cpu().numpy()

        names = batch["name"]

        shapes = batch["shape"]

        for i in range(len(names)):

            H = shapes[0][i].item()
            W = shapes[1][i].item()

            mask = pred_masks[i, 0]

            mask = cv2.resize(
                mask,
                (W, H),
                interpolation=cv2.INTER_NEAREST
            )

            json_name = names[i].replace(".png", ".json")

            save_path = os.path.join(
                save_dir,
                json_name
            )

            save_json(
                save_path,
                mask,
                (H, W)
            )


# =========================
# main
# =========================

def main(args):

    device = torch.device("cuda")

    model = MODELS.get(args.model)

    model = model(
        model_setting_name=args.model_setting_name,
        #edge_mask_width=args.edge_mask_width,
    )

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model"],
        strict=False
    )

    model.to(device)

    dataset = CompetitionDataset(
        args.test_image_dir,
        image_size=args.image_size
    )

    loader = DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    inference(
        model,
        loader,
        device,
        args.save_json_dir
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="MaskCLIP"
    )

    parser.add_argument(
        "--model_setting_name",
        type=str,
        default="ViTL"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--test_image_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--save_json_dir",
        type=str,
        default="./submission"
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=512
    )

    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8
    )

    parser.add_argument(
        "--edge_mask_width",
        type=int,
        default=7
    )

    args = parser.parse_args()

    main(args)