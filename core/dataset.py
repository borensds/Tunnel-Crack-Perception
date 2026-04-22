import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob


def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5
            ),

            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20),
            ], p=0.4),

            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(),
                A.MotionBlur(blur_limit=3),
            ], p=0.3),

            # ImageNet 均值方差归一化
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


class TunnelDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.valid_samples = self._check_and_build_dataset()

    def _check_and_build_dataset(self):
        valid_samples = []
        image_paths = glob(os.path.join(self.images_dir, '*.[jp][pn]g'))

        if not image_paths:
            print(f"警告：在 {self.images_dir} 目录下没有找到图像！")
            return valid_samples

        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path_png = os.path.join(self.masks_dir, base_name + '.png')
            mask_path_jpg = os.path.join(self.masks_dir, base_name + '.jpg')

            if os.path.exists(mask_path_png):
                final_mask_path = mask_path_png
            elif os.path.exists(mask_path_jpg):
                final_mask_path = mask_path_jpg
            else:
                final_mask_path = None

            valid_samples.append({
                'image': img_path,
                'mask': final_mask_path,
                'is_negative': final_mask_path is None
            })

        print(f"数据流构建完成：扫描到 {len(valid_samples)} 个可用样本。")
        return valid_samples

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, i):
        sample = self.valid_samples[i]

        image = cv2.imread(sample['image'])
        assert image is not None, f"读取图像失败 {sample['image']}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        if not sample['is_negative']:
            mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
            assert mask is not None, f"读取标签失败 {sample['mask']}"
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        is_negative_tensor = torch.tensor(1.0 if sample['is_negative'] else 0.0, dtype=torch.float32)
        return image, mask.float(), is_negative_tensor


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMG_DIR = os.path.join(ROOT_DIR, "dummy_data", "images")
    MASK_DIR = os.path.join(ROOT_DIR, "dummy_data", "masks")

    if os.path.exists(IMG_DIR):
        ds = TunnelDataset(IMG_DIR, MASK_DIR, transform=get_transforms('train'))
        print(f"Dataset test passed. Size: {len(ds)}")