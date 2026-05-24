import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import streamlit as st
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from model import SiameseChangeDetector
from inference import predict_patch


st.set_page_config(
    page_title="Satellite Change Detection",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Satellite Imagery Change Detection")
st.markdown(
    "Upload a **before** and **after** satellite image patch "
    "to detect changes — new construction, deforestation, "
    "infrastructure changes."
)

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

threshold = st.sidebar.slider(
    "Detection threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Higher = only confident changes flagged"
)

checkpoint_path = st.sidebar.text_input(
    "Model checkpoint path",
    value="models/best_model.pth"
)

use_demo = st.sidebar.checkbox(
    "Use demo mode (no checkpoint needed)",
    value=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Siamese ResNet-50")
st.sidebar.markdown("**Dataset:** LEVIR-CD")
st.sidebar.markdown("**Input size:** 256 × 256")
st.sidebar.markdown("**Loss:** BCE + Dice")
st.sidebar.markdown("**Encoder:** ResNet-50 (ImageNet)")


# ── Image upload ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("T1 — Before image")
    t1_file = st.file_uploader(
        "Upload before image",
        type=["png", "jpg", "jpeg"],
        key="t1"
    )
    if t1_file:
        t1_img = Image.open(t1_file).convert("RGB").resize((256, 256))
        st.image(t1_img, use_container_width=True)

with col2:
    st.subheader("T2 — After image")
    t2_file = st.file_uploader(
        "Upload after image",
        type=["png", "jpg", "jpeg"],
        key="t2"
    )
    if t2_file:
        t2_img = Image.open(t2_file).convert("RGB").resize((256, 256))
        st.image(t2_img, use_container_width=True)


# ── Run inference ─────────────────────────────────────────
st.markdown("---")
run = st.button("🔍 Detect Changes", type="primary", use_container_width=True)

if run:
    if not t1_file or not t2_file:
        st.error("⚠️ Please upload both a before and after image.")
        st.stop()

    with st.spinner("Running model inference..."):

        t1_arr = np.array(t1_img)
        t2_arr = np.array(t2_img)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if use_demo:
            st.info(
                "ℹ️ Demo mode active — showing simulated output. "
                "Train the model and uncheck demo mode for real predictions."
            )
            np.random.seed(42)
            base = np.random.rand(256, 256).astype(np.float32)
            prob_map = gaussian_filter(base, sigma=12)
            prob_map = (prob_map - prob_map.min()) / (
                prob_map.max() - prob_map.min()
            )

        else:
            if not os.path.exists(checkpoint_path):
                st.error(f"❌ Checkpoint not found: `{checkpoint_path}`")
                st.stop()

            model = SiameseChangeDetector(
                encoder_name="resnet50",
                pretrained=False
            ).to(device)

            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            prob_map = predict_patch(
                model, t1_arr, t2_arr, device, transform=None
            )

        change_map = (prob_map > threshold).astype(np.uint8)
        changed_pct = change_map.mean() * 100

    # ── Metrics ───────────────────────────────────────────
    st.subheader("📊 Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Changed pixels", f"{change_map.sum():,}")
    m2.metric("Changed area", f"{changed_pct:.1f}%")
    m3.metric("Threshold", f"{threshold:.2f}")
    m4.metric("Device", str(device).upper())

    st.markdown("---")

    # ── Visualisations ────────────────────────────────────
    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown("**🌡️ Change probability heatmap**")
        st.caption("Red = high change probability, Green = low")
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(prob_map, cmap="RdYlGn_r", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")
        st.pyplot(fig)
        plt.close()

    with r2:
        st.markdown("**⬛ Binary change mask**")
        st.caption("White = detected change, Black = no change")
        mask_display = (change_map * 255).astype(np.uint8)
        st.image(mask_display, use_container_width=True)

    with r3:
        st.markdown("**🔴 Change overlay on T2**")
        st.caption("Red regions = detected changes")
        t2_overlay = t2_arr.copy()
        t2_overlay[change_map == 1, 0] = 255
        t2_overlay[change_map == 1, 1] = 50
        t2_overlay[change_map == 1, 2] = 50
        st.image(t2_overlay, use_container_width=True)

    # ── Side by side comparison ───────────────────────────
    st.markdown("---")
    st.subheader("🔄 Before / After comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**T1 — Before**")
        st.image(t1_arr, use_container_width=True)
    with c2:
        st.markdown("**T2 — After**")
        st.image(t2_arr, use_container_width=True)

    # ── Download ──────────────────────────────────────────
    st.markdown("---")
    import io
    buf = io.BytesIO()
    Image.fromarray(mask_display).save(buf, format="PNG")

    st.download_button(
        label="⬇️ Download change mask",
        data=buf.getvalue(),
        file_name="change_mask.png",
        mime="image/png",
        use_container_width=True
    )

    # ── Summary ───────────────────────────────────────────
    if changed_pct > 30:
        st.warning(
            f"⚠️ High change detected: **{changed_pct:.1f}%** of the area "
            f"shows significant change between T1 and T2."
        )
    elif changed_pct > 10:
        st.info(
            f"ℹ️ Moderate change detected: **{changed_pct:.1f}%** of the "
            f"area shows change."
        )
    else:
        st.success(
            f"✅ Low change detected: only **{changed_pct:.1f}%** of the "
            f"area shows change."
        )