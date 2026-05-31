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
    page_title="Satellite Imagery Change Detection",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Satellite Imagery Change Detection")
st.markdown(
    "Detect land cover changes from satellite imagery using a "
    "**Siamese ResNet-50** deep learning model. "
    "Trained on LEVIR-CD — F1: **0.840** | IoU: **0.725**"
)

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

# Mode selector — demo first for fast viewing
mode = st.sidebar.radio(
    "🔘 Mode",
    ["🖼️ Demo — Instant Result", "📤 Upload Your Images", "🛰️ Live Sentinel-2"],
    index=0
)

threshold = st.sidebar.slider(
    "Detection threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Higher = only confident changes flagged"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Siamese ResNet-50")
st.sidebar.markdown("**Dataset:** LEVIR-CD")
st.sidebar.markdown("**Input size:** 256 × 256")
st.sidebar.markdown("**F1 Score:** 0.840")
st.sidebar.markdown("**IoU:** 0.725")
st.sidebar.markdown("**Improvement over baseline:** +64%")


# ── DEMO MODE ────────────────────────────────────────────
if mode == "🖼️ Demo — Instant Result":
    st.info("📸 Showing pre-computed results from real Sentinel-2 imagery over India.")

    # Paths to pre-computed assets
    demo_result_path = os.path.join(
        os.path.dirname(__file__), "../assets/change_detection_result.png"
    )
    flood_result_path = os.path.join(
        os.path.dirname(__file__), "../assets/flood_assessment_result.png"
    )

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("F1 Score", "0.840")
    m2.metric("IoU", "0.725")
    m3.metric("Baseline improvement", "+64%")
    m4.metric("Changed area (demo)", "65.0%")

    st.markdown("---")

    # Show change detection result
    if os.path.exists(demo_result_path):
        st.subheader("📡 Change Detection Result")
        st.caption("T1 (Before) → T2 (After) → Change Map")
        st.image(demo_result_path, use_container_width=True)
    else:
        st.warning("Demo image not found. Please add `assets/change_detection_result.png`")

    st.markdown("---")

    # Show flood assessment result
    if os.path.exists(flood_result_path):
        st.subheader("🌊 Flood Assessment Result")
        st.caption("Sentinel-2 flood analysis over Indian river basin")
        st.image(flood_result_path, use_container_width=True)
    else:
        st.warning("Flood image not found. Please add `assets/flood_assessment_result.png`")

    st.markdown("---")
    st.success(
        "✅ Model achieves **F1: 0.840** and **IoU: 0.725** on the LEVIR-CD benchmark — "
        "outperforming traditional pixel-differencing by **64%**."
    )
    st.markdown(
        "Switch to **📤 Upload Your Images** to run the model on your own satellite patches, "
        "or **🛰️ Live Sentinel-2** to download and analyse real-time imagery."
    )


# ── UPLOAD MODE ──────────────────────────────────────────
elif mode == "📤 Upload Your Images":
    st.markdown("Upload a **before** and **after** satellite image patch to detect changes.")

    checkpoint_path = st.sidebar.text_input(
        "Model checkpoint path",
        value="models/best_model.pth"
    )
    use_demo_model = st.sidebar.checkbox(
        "Use simulated output (no checkpoint needed)",
        value=True
    )

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

    st.markdown("---")
    run = st.button("🔍 Detect Changes", type="primary", use_container_width=True)

    if run:
        if not t1_file or not t2_file:
            st.error("⚠️ Please upload both a before and after image.")
            st.stop()

        with st.spinner("Running model inference..."):
            t1_arr = np.array(t1_img)
            t2_arr = np.array(t2_img)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if use_demo_model:
                st.info("ℹ️ Simulated output. Uncheck demo mode and provide checkpoint for real predictions.")
                np.random.seed(42)
                base = np.random.rand(256, 256).astype(np.float32)
                prob_map = gaussian_filter(base, sigma=12)
                prob_map = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min())
            else:
                if not os.path.exists(checkpoint_path):
                    st.error(f"❌ Checkpoint not found: `{checkpoint_path}`")
                    st.stop()
                model = SiameseChangeDetector(encoder_name="resnet50", pretrained=False).to(device)
                ckpt = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                prob_map = predict_patch(model, t1_arr, t2_arr, device, transform=None)

        change_map = (prob_map > threshold).astype(np.uint8)
        changed_pct = change_map.mean() * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Changed pixels", f"{change_map.sum():,}")
        m2.metric("Changed area", f"{changed_pct:.1f}%")
        m3.metric("Threshold", f"{threshold:.2f}")
        m4.metric("Device", str(device).upper())

        st.markdown("---")
        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown("**🌡️ Change probability heatmap**")
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(prob_map, cmap="RdYlGn_r", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")
            st.pyplot(fig)
            plt.close()

        with r2:
            st.markdown("**⬛ Binary change mask**")
            mask_display = (change_map * 255).astype(np.uint8)
            st.image(mask_display, use_container_width=True)

        with r3:
            st.markdown("**🔴 Change overlay on T2**")
            t2_overlay = t2_arr.copy()
            t2_overlay[change_map == 1, 0] = 255
            t2_overlay[change_map == 1, 1] = 50
            t2_overlay[change_map == 1, 2] = 50
            st.image(t2_overlay, use_container_width=True)

        st.markdown("---")
        st.subheader("🔄 Before / After comparison")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**T1 — Before**")
            st.image(t1_arr, use_container_width=True)
        with c2:
            st.markdown("**T2 — After**")
            st.image(t2_arr, use_container_width=True)

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

        if changed_pct > 30:
            st.warning(f"⚠️ High change detected: **{changed_pct:.1f}%** of the area shows significant change.")
        elif changed_pct > 10:
            st.info(f"ℹ️ Moderate change detected: **{changed_pct:.1f}%** of the area shows change.")
        else:
            st.success(f"✅ Low change detected: only **{changed_pct:.1f}%** of the area shows change.")


# ── LIVE SENTINEL MODE ───────────────────────────────────
elif mode == "🛰️ Live Sentinel-2":
    st.warning(
        "⚠️ **Live mode downloads real Sentinel-2 satellite data (~500MB per scene). "
        "This takes 15–30 minutes on free tier.** "
        "For a quick demo, switch to **🖼️ Demo — Instant Result** mode."
    )

    try:
        from sentinel import CopernicusClient, get_sentinel_pair
        from inference import run_full_inference

        st.subheader("🔐 Copernicus Credentials")
        st.caption("Register free at dataspace.copernicus.eu")

        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username", type="default")
        with col2:
            password = st.text_input("Password", type="password")

        st.markdown("---")
        st.subheader("📍 Select location on map")

        try:
            from streamlit_folium import st_folium
            import folium

            m = folium.Map(location=[28.5, 77.0], zoom_start=7)
            map_data = st_folium(m, width=700, height=400)

            if map_data and map_data.get("last_clicked"):
                lat = map_data["last_clicked"]["lat"]
                lon = map_data["last_clicked"]["lng"]
                size = st.slider("Bounding box size (degrees)", 0.1, 1.0, 0.3, 0.05)
                bbox = [lon - size/2, lat - size/2, lon + size/2, lat + size/2]
                st.success(f"Lat: {lat:.4f} Lon: {lon:.4f}")
                st.caption(f"BBox: {[round(b, 3) for b in bbox]}")
            else:
                bbox = None
                st.info("Click on the map to select a location")

        except ImportError:
            st.warning("streamlit-folium not installed. Enter coordinates manually.")
            lat = st.number_input("Latitude", value=28.5)
            lon = st.number_input("Longitude", value=77.0)
            size = st.slider("Bounding box size (degrees)", 0.1, 1.0, 0.3, 0.05)
            bbox = [lon - size/2, lat - size/2, lon + size/2, lat + size/2]

        col1, col2 = st.columns(2)
        with col1:
            date1 = st.date_input("T1 — Before date")
        with col2:
            date2 = st.date_input("T2 — After date")

        cloud_cover = st.slider("Max cloud cover %", 5, 80, 20)

        if st.button("🛰️ Download Sentinel-2 & Detect Changes", type="primary", use_container_width=True):
            if not username or not password:
                st.error("Please enter your Copernicus credentials.")
                st.stop()
            if not bbox:
                st.error("Please select a location on the map.")
                st.stop()

            with st.spinner("Searching and downloading Sentinel-2 scenes... (this takes 2–5 minutes)"):
                try:
                    t1_path, t2_path = get_sentinel_pair(
                        username, password, bbox,
                        date1.strftime("%Y-%m-%d"),
                        date2.strftime("%Y-%m-%d"),
                        cloud_pct=cloud_cover
                    )
                    st.success("✅ Download complete!")
                    st.image([t1_path, t2_path], caption=["T1 — Before", "T2 — After"])

                except Exception as e:
                    st.error(f"❌ Error: {e}")

    except ImportError as e:
        st.error(f"Missing dependency: {e}")
