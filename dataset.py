import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from tqdm import tqdm

def copy_paste(image, mask, paste_image, paste_mask, alpha=1.0):
    result_image = image.copy()
    result_mask  = mask.copy()
    for cls in [1, 2]:
        obj_pixels = (paste_mask == cls)
        if obj_pixels.sum() == 0: continue
        dx, dy = random.randint(-30, 30), random.randint(-30, 30)
        M  = np.float32([[1, 0, dx], [0, 1, dy]])
        h, w = image.shape[:2]
        shifted_mask  = cv2.warpAffine(obj_pixels.astype(np.uint8), M, (w, h))
        shifted_image = cv2.warpAffine(paste_image, M, (w, h))
        bin_mask = shifted_mask.astype(bool)
        result_image[bin_mask] = (alpha * shifted_image[bin_mask] + (1 - alpha) * result_image[bin_mask]).astype(np.uint8)
        result_mask[bin_mask] = cls
    return result_image, result_mask

aug_flip = v2.Compose([v2.RandomHorizontalFlip(p=1.0), v2.RandomVerticalFlip(p=1.0)])
aug_rotate = v2.RandomRotation(degrees=[-90, 90])
aug_combo_1 = v2.Compose([aug_flip, aug_rotate])

class SugarBeetsTileDataset(Dataset):
    def __init__(self, root_dir, split_file, tile_size=256, overlap=0, transform=None, use_copy_paste=False, max_samples=None):
        self.root_dir = root_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transform
        self.use_copy_paste = use_copy_paste
        self.max_samples = max_samples

        with open(os.path.join(root_dir, split_file), 'r') as f:
            self.mask_names = f.read().splitlines()

        if len(self.mask_names) == 0:
            raise ValueError(f"Split file is empty: {split_file}")

        self.cache = {}
        self.tiles = []

        # Load only the first sample to infer image size and build tile coordinates.
        first_mask_filename = self.mask_names[0]
        first_img, first_mask = self._load_sample(first_mask_filename)
        self.cache[first_mask_filename] = (first_img, first_mask)

        stride = self.tile_size - self.overlap
        H, W = first_img.shape[:2]
        mask_iter = tqdm(self.mask_names, desc="Building tiles", unit="mask") if tqdm else self.mask_names
        for m_name in mask_iter:
            for y in range(0, H - self.tile_size + 1, stride):
                for x in range(0, W - self.tile_size + 1, stride):
                    self.tiles.append((m_name, y, x))

        if self.max_samples is not None and len(self.tiles) > self.max_samples:
            self.tiles = random.sample(self.tiles, self.max_samples)

    def _load_sample(self, mask_filename):
        base_num = mask_filename.split('.')[0]
        img_path = os.path.join(self.root_dir, 'rgb', f"rgb_{int(base_num):05d}.png")
        mask_path = os.path.join(self.root_dir, 'annotations', mask_filename)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask_bgr = cv2.imread(mask_path)
        if mask_bgr is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

        mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
        mask[(mask_rgb == [255, 0, 0]).all(axis=2)] = 1
        mask[(mask_rgb == [0, 0, 255]).all(axis=2)] = 2

        return img, mask

    def __len__(self): return len(self.tiles)

    def __getitem__(self, idx):
        mask_filename, y, x = self.tiles[idx]
        t = self.tile_size

        if mask_filename not in self.cache:
            self.cache[mask_filename] = self._load_sample(mask_filename)
        img, mask = self.cache[mask_filename]
        t_img, t_mask = img[y:y+t, x:x+t].copy(), mask[y:y+t, x:x+t].copy()

        if self.use_copy_paste and random.random() < 0.5:
            d_filename, dy, dx = self.tiles[random.randint(0, len(self.tiles) - 1)]
            if d_filename not in self.cache:
                self.cache[d_filename] = self._load_sample(d_filename)
            d_img, d_mask = self.cache[d_filename]
            p_img, p_mask = d_img[dy:dy+t, dx:dx+t].copy(), d_mask[dy:dy+t, dx:dx+t].copy()
            t_img, t_mask = copy_paste(t_img, t_mask, p_img, p_mask, alpha=1.0)

        t_img = np.transpose(t_img.astype(np.float32) / 255.0, (2, 0, 1))
        img_t, mask_t = torch.tensor(t_img), torch.tensor(t_mask, dtype=torch.long)

        if self.transform:
            mask_t = tv_tensors.Mask(mask_t)
            img_t, mask_t = self.transform(img_t, mask_t)

        return img_t, torch.tensor(np.array(mask_t), dtype=torch.long)