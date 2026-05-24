import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(split="train", image_size=256):
    if split == "train":
        return A.ReplayCompose([
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ], additional_targets={"image2": "image"})
    else:
        return A.ReplayCompose([
            A.CenterCrop(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ], additional_targets={"image2": "image"})


class LEVIRDataset(Dataset):
    """
    LEVIR-CD dataset loader.

    Expected folder structure:
        data/LEVIR-CD/
            train/
                A/        <- T1 images (before)
                B/        <- T2 images (after)
                label/    <- binary change masks
            val/
                A/
                B/
                label/
            test/
                A/
                B/
                label/

    Images are RGB PNG, 1024x1024.
    Labels are grayscale PNG: 255 = change, 0 = no change.
    """

    def __init__(self, root_dir, split="train", image_size=256):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.transform = get_transforms(split, image_size)

        self.t1_dir = os.path.join(root_dir, split, "A")
        self.t2_dir = os.path.join(root_dir, split, "B")
        self.mask_dir = os.path.join(root_dir, split, "label")

        self.filenames = sorted([
            f for f in os.listdir(self.t1_dir)
            if f.endswith(".png") or f.endswith(".jpg")
        ])

        print(f"[{split}] Found {len(self.filenames)} image pairs")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        t1 = np.array(Image.open(
            os.path.join(self.t1_dir, fname)).convert("RGB"))
        t2 = np.array(Image.open(
            os.path.join(self.t2_dir, fname)).convert("RGB"))
        mask = np.array(Image.open(
            os.path.join(self.mask_dir, fname)).convert("L"))

        # Binarise mask: 255 -> 1, 0 -> 0
        mask = (mask > 127).astype(np.float32)

        # Apply SAME spatial transforms to both images
        augmented = self.transform(image=t1, image2=t2, mask=mask)
        t1_tensor = augmented["image"]
        t2_tensor = augmented["image2"]
        mask_tensor = augmented["mask"].unsqueeze(0)  # [1, H, W]

        return {
            "t1": t1_tensor,        # [3, H, W] float32
            "t2": t2_tensor,        # [3, H, W] float32
            "mask": mask_tensor,    # [1, H, W] float32 binary
            "filename": fname
        }


if __name__ == "__main__":
    # Quick sanity check — run with: python src/dataset.py
    dataset = LEVIRDataset(
        root_dir="data/LEVIR-CD",
        split="train",
        image_size=256
    )
    sample = dataset[0]
    print("T1 shape:  ", sample["t1"].shape)
    print("T2 shape:  ", sample["t2"].shape)
    print("Mask shape:", sample["mask"].shape)
    print("Mask unique values:", sample["mask"].unique())
    print("File:", sample["filename"])