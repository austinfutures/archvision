import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class PlanDataset(Dataset):
    def __init__(self, img_dir="data/images", mask_dir="data/masks", augment=False, indices=None):
        self.imgs = sorted(glob.glob(f"{img_dir}/*.png"))
        self.masks = sorted(glob.glob(f"{mask_dir}/*.png"))
        if indices is not None:
            self.imgs = [self.imgs[i] for i in indices]
            self.masks = [self.masks[i] for i in indices]
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = cv2.imread(self.imgs[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)

        if self.augment:
            img, mask = self._augment(img, mask)

        img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.int64))
        return img, mask

    @staticmethod
    def _augment(img, mask):
        if random.random() < 0.5:
            img, mask = np.fliplr(img), np.fliplr(mask)
        if random.random() < 0.5:
            img, mask = np.flipud(img), np.flipud(mask)
        k = random.randint(0, 3)
        img, mask = np.rot90(img, k), np.rot90(mask, k)
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-15, 15)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)
