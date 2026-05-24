import os
import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import SiameseChangeDetector


def get_inference_transform(image_size=256):
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def load_model(checkpoint_path, encoder_name="resnet50", device="cpu"):
    """Load trained model from checkpoint."""
    model = SiameseChangeDetector(
        encoder_name=encoder_name,
        pretrained=False
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"— val F1: {checkpoint['val_f1']:.4f}")
    return model


def predict_patch(model, t1_arr, t2_arr, device, transform):
    """
    Run inference on a single numpy image pair.

    Args:
        t1_arr: np.ndarray [H, W, 3] uint8
        t2_arr: np.ndarray [H, W, 3] uint8

    Returns:
        prob_map: np.ndarray [H, W] float32, values in [0, 1]
    """
    aug = transform(image=t1_arr, image2=t2_arr)

    # albumentations ReplayCompose not needed here — just basic transform
    t1_aug = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])(image=t1_arr)["image"]

    t2_aug = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])(image=t2_arr)["image"]

    t1_tensor = t1_aug.unsqueeze(0).to(device)   # [1, 3, H, W]
    t2_tensor = t2_aug.unsqueeze(0).to(device)   # [1, 3, H, W]

    with torch.no_grad():
        logits = model(t1_tensor, t2_tensor)
        prob = torch.sigmoid(logits)      # [1, 1, H, W]

    return prob.squeeze().cpu().numpy()           # [H, W]


def sliding_window_inference(model, t1_path, t2_path,
                             patch_size=256, stride=128,
                             threshold=0.5, device="cpu"):
    """
    Run inference over full-size images using sliding window.
    Overlapping patches are averaged for smoother predictions.

    Args:
        t1_path: path to T1 image (before)
        t2_path: path to T2 image (after)
        patch_size: size of each patch
        stride: step between patches (overlap = patch_size - stride)
        threshold: binarisation threshold

    Returns:
        prob_map:   np.ndarray [H, W] float32
        change_map: np.ndarray [H, W] uint8 binary (0 or 255)
    """
    t1_img = np.array(Image.open(t1_path).convert("RGB"))
    t2_img = np.array(Image.open(t2_path).convert("RGB"))

    H, W, _ = t1_img.shape
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    y_starts = list(range(0, H - patch_size + 1, stride))
    x_starts = list(range(0, W - patch_size + 1, stride))

    # Make sure we cover the edges
    if y_starts[-1] + patch_size < H:
        y_starts.append(H - patch_size)
    if x_starts[-1] + patch_size < W:
        x_starts.append(W - patch_size)

    total = len(y_starts) * len(x_starts)
    done = 0

    print(f"Running sliding window inference...")
    print(f"Image: {H}x{W} | Patch: {patch_size} | Stride: {stride}")
    print(f"Total patches: {total}")

    for y in y_starts:
        for x in x_starts:
            t1_patch = t1_img[y:y+patch_size, x:x+patch_size]
            t2_patch = t2_img[y:y+patch_size, x:x+patch_size]

            prob = predict_patch(
                model, t1_patch, t2_patch,
                device, transform=None
            )

            prob_map[y:y+patch_size, x:x+patch_size] += prob
            count_map[y:y+patch_size, x:x+patch_size] += 1.0

            done += 1
            if done % 20 == 0:
                print(f"  {done}/{total} patches done")

    # Average overlapping predictions
    count_map = np.maximum(count_map, 1)
    prob_map = prob_map / count_map

    # Binarise
    change_map = (prob_map > threshold).astype(np.uint8) * 255

    return prob_map, change_map


def visualise_results(t1_path, t2_path, prob_map,
                      change_map, save_path=None):
    """
    Plot T1 | T2 | Change heatmap | Binary mask side by side.
    """
    t1 = np.array(Image.open(t1_path).convert("RGB"))
    t2 = np.array(Image.open(t2_path).convert("RGB"))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Satellite Change Detection Results", fontsize=14)

    axes[0].imshow(t1)
    axes[0].set_title("T1 — Before")
    axes[0].axis("off")

    axes[1].imshow(t2)
    axes[1].set_title("T2 — After")
    axes[1].axis("off")

    im = axes[2].imshow(prob_map, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[2].set_title("Change probability")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(change_map, cmap="gray")
    axes[3].set_title("Change mask (binary)")
    axes[3].axis("off")

    changed_pct = (change_map > 0).mean() * 100
    fig.text(0.5, 0.01,
             f"Changed area: {changed_pct:.2f}% of image",
             ha="center", fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualisation to {save_path}")
    else:
        plt.show()

    plt.close()


def predict_pair(t1_path, t2_path, checkpoint_path,
                 output_dir="outputs", threshold=0.5,
                 patch_size=256, stride=128):
    """
    Full inference pipeline for a single image pair.
    Saves probability map, binary mask, and visualisation.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path, device=device)

    prob_map, change_map = sliding_window_inference(
        model, t1_path, t2_path,
        patch_size=patch_size,
        stride=stride,
        threshold=threshold,
        device=device
    )

    # Save outputs
    stem = os.path.splitext(os.path.basename(t1_path))[0]

    prob_save = os.path.join(output_dir, f"{stem}_prob.npy")
    np.save(prob_save, prob_map)

    mask_save = os.path.join(output_dir, f"{stem}_mask.png")
    Image.fromarray(change_map).save(mask_save)

    viz_save = os.path.join(output_dir, f"{stem}_viz.png")
    visualise_results(t1_path, t2_path, prob_map, change_map,
                      save_path=viz_save)

    changed_pct = (change_map > 0).mean() * 100
    print(f"\nResults saved to {output_dir}/")
    print(f"Changed area: {changed_pct:.2f}%")

    return prob_map, change_map


if __name__ == "__main__":
    # Quick test with dummy images — no checkpoint needed
    # Replace paths with real images once you have the dataset
    import sys

    print("Inference module loaded successfully.")
    print("Usage:")
    print("  from inference import predict_pair")
    print("  predict_pair('t1.png', 't2.png', 'models/best_model.pth')")