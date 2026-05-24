import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import wandb

from dataset import LEVIRDataset
from model import SiameseChangeDetector, CombinedLoss


def iou_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def f1_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6)


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss = 0
    total_f1 = 0
    total_iou = 0

    loop = tqdm(loader, desc="Train", leave=False)
    for batch in loop:
        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        with autocast():
            pred = model(t1, t2)
            loss = loss_fn(pred, mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        f1 = f1_score(pred.detach(), mask)
        iou = iou_score(pred.detach(), mask)

        total_loss += loss.item()
        total_f1 += f1.item()
        total_iou += iou.item()

        loop.set_postfix(loss=f"{loss.item():.4f}", f1=f"{f1.item():.4f}")

    n = len(loader)
    return total_loss / n, total_f1 / n, total_iou / n


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_f1 = 0
    total_iou = 0

    loop = tqdm(loader, desc="Val", leave=False)
    for batch in loop:
        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)
        mask = batch["mask"].to(device)

        pred = model(t1, t2)
        loss = loss_fn(pred, mask)

        f1 = f1_score(pred, mask)
        iou = iou_score(pred, mask)

        total_loss += loss.item()
        total_f1 += f1.item()
        total_iou += iou.item()

    n = len(loader)
    return total_loss / n, total_f1 / n, total_iou / n


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # --- Datasets ---
    train_dataset = LEVIRDataset(
        root_dir=config["data_dir"],
        split="train",
        image_size=config["image_size"]
    )
    val_dataset = LEVIRDataset(
        root_dir=config["data_dir"],
        split="val",
        image_size=config["image_size"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # --- Model ---
    model = SiameseChangeDetector(
        encoder_name=config["encoder"],
        pretrained=True
    ).to(device)

    loss_fn = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=1e-6
    )
    scaler = GradScaler()

    # --- WandB (optional) ---
    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        wandb.init(
            project="satellite-change-detection",
            config=config
        )

    # --- Training loop ---
    best_val_f1 = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")

        train_loss, train_f1, train_iou = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device)

        val_loss, val_f1, val_iou = validate(
            model, val_loader, loss_fn, device)

        scheduler.step()

        print(f"  Train — loss: {train_loss:.4f}  f1: {train_f1:.4f}  iou: {train_iou:.4f}")
        print(f"  Val   — loss: {val_loss:.4f}  f1: {val_f1:.4f}  iou: {val_iou:.4f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/f1": train_f1,
                "train/iou": train_iou,
                "val/loss": val_loss,
                "val/f1": val_f1,
                "val/iou": val_iou,
                "lr": scheduler.get_last_lr()[0]
            })

        # Save best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_iou": val_iou,
                "config": config
            }
            torch.save(checkpoint, "models/best_model.pth")
            print(f"  Saved best model — val F1: {val_f1:.4f}")

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    config = {
        "data_dir":    "data/LEVIR-CD",
        "image_size":  256,
        "batch_size":  8,
        "num_workers": 2,
        "encoder":     "resnet50",
        "lr":          1e-4,
        "epochs":      50,
        "use_wandb":   False,   # set True once you have wandb account
    }
    train(config)