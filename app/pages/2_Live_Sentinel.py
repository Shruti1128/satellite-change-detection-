import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
from PIL import Image
import torch

from model import SiameseChangeDetector
from inference import predict_patch
from sentinel import get_sentinel_pair

st.set_page_config(
    page_title="Live Sentinel-2 Change Detection",
    page_icon="🛰️",
    layout="wide"
)

st.title("🌍 Live Sentinel-2 Change Detection")
st.markdown(
    "Select any location on Earth, pick two dates, "
    "and detect real changes using Sentinel-2 satellite imagery."
)

# ── Credentials ───────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**Copernicus credentials**")
    username = st.text_input("Username", type="default")
    password = st.text_input("Password", type="password")
    st.markdown("Register free at [dataspace.copernicus.eu](https://dataspace.copernicus.eu)")
    st.markdown("---")
    date1 = st.date_input("T1 — Before date")
    date2 = st.date_input("T2 — After date")
    cloud_pct = st.slider("Max cloud cover %", 0, 50, 20)
    threshold = st.slider("Detection threshold", 0.1, 0.9, 0.5, 0.05)
    checkpoint = st.text_input("Checkpoint path", "models/best_model.pth")
    use_demo = st.checkbox("Demo mode (no checkpoint)", value=True)

# ── Map ───────────────────────────────────────────────────
st.subheader("📍 Step 1 — Select location on map")
st.caption("Click the map to set centre, then adjust the bounding box size below")

col1, col2 = st.columns([3, 1])

with col1:
    m = folium.Map(location=[28.6, 77.2], zoom_start=8)
    folium.TileLayer("CartoDB positron").add_to(m)

    map_data = st_folium(m, height=400, width=700)

with col2:
    st.markdown("**Bounding box size**")
    box_size = st.slider("Degrees", 0.1, 2.0, 0.3, 0.1)
    st.markdown("**Selected location**")

    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        st.success(f"Lat: {lat:.4f}\nLon: {lon:.4f}")
        bbox = [
            lon - box_size/2, lat - box_size/2,
            lon + box_size/2, lat + box_size/2
        ]
        st.caption(f"BBox: {[round(b,3) for b in bbox]}")
    else:
        st.info("Click anywhere on the map")
        # Default: Delhi
        lat, lon = 28.6, 77.2
        bbox = [76.85, 28.35, 77.15, 28.65]

# ── Download & Detect ─────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Step 2 — Download & detect changes")

if st.button("Download Sentinel-2 & Detect Changes",
             type="primary", use_container_width=True):

    if not username or not password:
        st.warning("Enter your Copernicus credentials in the sidebar.")
        st.stop()

    if str(date1) >= str(date2):
        st.error("T2 (after) date must be later than T1 (before) date.")
        st.stop()

    with st.spinner(f"Searching and downloading Sentinel-2 scenes... "
                    f"(this takes 2–5 minutes)"):
        try:
            t1_path, t2_path = get_sentinel_pair(
                username=username,
                password=password,
                bbox=bbox,
                date1=str(date1),
                date2=str(date2),
                cloud_pct=cloud_pct
            )
            st.success("Downloaded real Sentinel-2 imagery ✅")

        except Exception as e:
            st.error(f"Download failed: {e}")
            st.info("Try increasing the cloud cover % or changing dates.")
            st.stop()

    # Load images
    t1_img = np.array(Image.open(t1_path).convert("RGB").resize((256, 256)))
    t2_img = np.array(Image.open(t2_path).convert("RGB").resize((256, 256)))

    # Run model
    with st.spinner("Running change detection model..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if use_demo:
            from scipy.ndimage import gaussian_filter
            np.random.seed(42)
            prob_map = gaussian_filter(
                np.random.rand(256, 256).astype(np.float32), sigma=12)
            prob_map = (prob_map - prob_map.min()) / (
                prob_map.max() - prob_map.min())
        else:
            model = SiameseChangeDetector("resnet50", pretrained=False).to(device)
            ckpt = torch.load(checkpoint, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            logits = predict_patch(model, t1_img, t2_img, device, None)
            prob_map = torch.sigmoid(torch.tensor(logits)).numpy()

        change_map = (prob_map > threshold).astype(np.uint8)
        changed_pct = change_map.mean() * 100

    # Results
    st.subheader("📊 Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Changed area", f"{changed_pct:.1f}%")
    m2.metric("Location", f"{lat:.3f}, {lon:.3f}")
    m3.metric("Date range", f"{date1} → {date2}")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("**T1 — Before**")
        st.image(t1_img, use_container_width=True)
    with r2:
        st.markdown("**T2 — After**")
        st.image(t2_img, use_container_width=True)
    with r3:
        st.markdown("**Change overlay**")
        overlay = t2_img.copy()
        overlay[change_map == 1, 0] = 255
        overlay[change_map == 1, 1] = 50
        overlay[change_map == 1, 2] = 50
        st.image(overlay, use_container_width=True)

    # Results map
    st.subheader("🗺️ Change map on location")
    result_map = folium.Map(location=[lat, lon], zoom_start=11)
    folium.Rectangle(
        bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
        color="red", fill=True, fill_opacity=0.2,
        popup=f"Changed area: {changed_pct:.1f}%"
    ).add_to(result_map)
    folium.Marker(
        [lat, lon],
        popup=f"Change detected: {changed_pct:.1f}%",
        icon=folium.Icon(color="red" if changed_pct > 10 else "green")
    ).add_to(result_map)
    st_folium(result_map, height=400, use_container_width=True)