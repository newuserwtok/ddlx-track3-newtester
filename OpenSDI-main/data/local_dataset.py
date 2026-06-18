import os
import cv2
import random
import numpy as np

import torch
from torch.utils.data import Dataset


class LocalDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        output_size=(512, 512),
        common_transforms=None,
        post_transform=None,
        post_transform_sam=None,
        edge_width=7,

        sample_ratio=1.0,
        max_samples=None,
        seed=42,
    ):

        self.root_dir = root_dir
        self.split = split

        self.output_size = output_size

        self.common_transforms = common_transforms
        self.post_transform = post_transform
        self.post_transform_sam = post_transform_sam

        self.edge_width = edge_width

        split_dir = os.path.join(root_dir, split)

        self.fake_dir = os.path.join(split_dir, "fake")
        self.mask_dir = os.path.join(split_dir, "masks")
        self.real_dir = os.path.join(split_dir, "real")

        self.samples = []

        # fake
        fake_files = sorted(os.listdir(self.fake_dir))

        for fname in fake_files:

            self.samples.append({
                "image_path": os.path.join(self.fake_dir, fname),
                "mask_path": os.path.join(self.mask_dir, fname),
                "label": 1,
                "name": fname
            })

        # real
        real_files = sorted(os.listdir(self.real_dir))

        for fname in real_files:

            self.samples.append({
                "image_path": os.path.join(self.real_dir, fname),
                "mask_path": None,
                "label": 0,
                "name": fname
            })

        # shuffle
        random.seed(seed)
        random.shuffle(self.samples)

        # sample ratio
        if sample_ratio < 1.0:

            keep_num = int(len(self.samples) * sample_ratio)

            self.samples = self.samples[:keep_num]

        # max samples
        if max_samples is not None:

            self.samples = self.samples[:max_samples]

        print("=" * 50)
        print(f"Dataset Split: {split}")
        print(f"Total Samples After Sampling: {len(self.samples)}")
        print("=" * 50)

    def __len__(self):
        return len(self.samples)

    def load_mask(self, path, h, w):

        if path is None:
            return np.zeros((h, w), dtype=np.uint8)

        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        mask = (mask > 127).astype(np.uint8)

        return mask

    def generate_edge_mask(self, mask):

        if mask.sum() == 0:
            return np.zeros_like(mask).astype(np.uint8)

        kernel = np.ones((self.edge_width, self.edge_width), np.uint8)

        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)

        edge = dilated - eroded

        edge = (edge > 0).astype(np.uint8)

        return edge

    def __getitem__(self, idx):

        item = self.samples[idx]

        image_path = item["image_path"]
        mask_path = item["mask_path"]

        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        mask = self.load_mask(mask_path, h, w)

        edge_mask = self.generate_edge_mask(mask)

        if self.common_transforms is not None:

            transformed = self.common_transforms(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"]

            edge_transformed = self.common_transforms(
                image=np.zeros_like(image),
                mask=edge_mask
            )

            edge_mask = edge_transformed["mask"]

        if self.post_transform is not None:

            transformed = self.post_transform(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"]

            edge_transformed = self.post_transform(
                image=np.zeros((h, w, 3), dtype=np.uint8),
                mask=edge_mask
            )

            edge_mask = edge_transformed["mask"]

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        if len(edge_mask.shape) == 2:
            edge_mask = edge_mask.unsqueeze(0)

        mask = mask.float()
        edge_mask = edge_mask.float()

        label = torch.tensor(item["label"]).long()

        return {
            "image": image,
            "mask": mask,
            "edge_mask": edge_mask,
            "label": label,
            "name": item["name"],
            "shape": (h, w)
        }