import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DoubleConv(nn.Module):
    """Two consecutive conv-bn-relu blocks used in the decoder."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SiameseChangeDetector(nn.Module):
    """
    Siamese network for binary change detection.

    Architecture:
        - Shared ResNet-50 encoder (ImageNet pretrained)
        - Feature difference |F1 - F2| at each scale
        - UNet-style decoder with skip connections
        - Final 1-channel sigmoid output (change probability map)

    Input:  t1, t2  — both [B, 3, H, W]
    Output: change probability map [B, 1, H, W], values in [0, 1]
    """

    def __init__(self, encoder_name="resnet50", pretrained=True):
        super().__init__()

        weights = "imagenet" if pretrained else None

        # Single encoder — called twice with T1 and T2 (weights shared)
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=weights,
        )

        # Encoder output channels per stage for resnet50:
        # [3, 64, 256, 512, 1024, 2048]
        enc_channels = self.encoder.out_channels

        # Decoder — takes difference features bottom-up
        self.decoder4 = DoubleConv(enc_channels[5], 256)
        self.decoder3 = DoubleConv(256 + enc_channels[4], 128)
        self.decoder2 = DoubleConv(128 + enc_channels[3], 64)
        self.decoder1 = DoubleConv(64  + enc_channels[2], 32)
        self.decoder0 = DoubleConv(32  + enc_channels[1], 16)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False)

        # Final head: 16 channels -> 1 channel probability map
        self.head = nn.Conv2d(16, 1, kernel_size=1)
        
    def encode(self, x):
        """Run encoder and return list of feature maps at each stage."""
        features = self.encoder(x)
        return features  # list of 6 tensors

    def forward(self, t1, t2):
        # --- Encode both images with shared weights ---
        f1 = self.encode(t1)   # [f0, f1, f2, f3, f4, f5]
        f2 = self.encode(t2)

        # --- Difference features at each scale ---
        # |F1 - F2| captures what changed, sign-invariant
        d = [torch.abs(a - b) for a, b in zip(f1, f2)]
        # d[0] = stem diff  (H/2,  64ch)
        # d[1] = stage1 diff(H/4,  256ch)
        # d[2] = stage2 diff(H/8,  512ch)
        # d[3] = stage3 diff(H/16, 1024ch)
        # d[4] = stage4 diff(H/32, 2048ch)
        # d[5] = stem/pool  (H/4,  64ch)  -- encoder includes stem

        # --- Decoder (bottom-up with skip connections) ---
        x = self.upsample(self.decoder4(d[5]))          # H/16
        x = self.upsample(self.decoder3(
            torch.cat([x, d[4]], dim=1)))               # H/8
        x = self.upsample(self.decoder2(
            torch.cat([x, d[3]], dim=1)))               # H/4
        x = self.upsample(self.decoder1(
            torch.cat([x, d[2]], dim=1)))               # H/2
        x = self.upsample(self.decoder0(
            torch.cat([x, d[1]], dim=1)))               # H

        return self.head(x)   # [B, 1, H, W]


class CombinedLoss(nn.Module):
    """
    BCE + Dice loss.
    Handles severe class imbalance (change pixels ~3-7% of dataset).

    loss = 0.5 * BCE + 0.5 * Dice
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def dice_loss(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1 - dice

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


if __name__ == "__main__":
    # Sanity check — run with: python src/model.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SiameseChangeDetector(
        encoder_name="resnet50",
        pretrained=False   # skip download for quick check
    ).to(device)

    t1 = torch.randn(2, 3, 256, 256).to(device)
    t2 = torch.randn(2, 3, 256, 256).to(device)

    with torch.no_grad():
        out = model(t1, t2)

    print(f"Input shape:  {t1.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")

    loss_fn = CombinedLoss()
    target = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)
    loss = loss_fn(out, target)
    print(f"Loss value:   {loss.item():.4f}")
    print("Model check passed.")