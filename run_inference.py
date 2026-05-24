import sys
sys.path.append("src")
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import SiameseChangeDetector
from inference import predict_patch

t1 = np.array(Image.open("data/sentinel/t1_rgb.png").convert("RGB").resize((256,256)))
t2 = np.array(Image.open("data/sentinel/t2_rgb.png").convert("RGB").resize((256,256)))

device = torch.device("cpu")
model = SiameseChangeDetector("resnet50", pretrained=False).to(device)
ckpt = torch.load("models/best_model.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

logits = predict_patch(model, t1, t2, device, None)
prob = torch.sigmoid(torch.tensor(logits)).numpy()
prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-6)

change = (prob > 0.5).astype(np.uint8)
change = (prob > 0.3).astype(np.uint8)
print(f"Changed area: {change.mean()*100:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(15,5))
axes[0].imshow(t1)
axes[0].set_title("T1 Apr 2026")
axes[0].axis("off")
axes[1].imshow(t2)
axes[1].set_title("T2 May 2026")
axes[1].axis("off")
axes[2].imshow(prob, cmap="RdYlGn_r", vmin=0, vmax=1)
axes[2].set_title("Change map")
axes[2].axis("off")
plt.savefig("data/sentinel/result.png", dpi=150, bbox_inches="tight")
print("Saved!")