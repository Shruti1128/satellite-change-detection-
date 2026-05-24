import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import json

from flood_monitor import (
    INDIAN_BASINS,
    calculate_flood_metrics,
    generate_flood_map,
    export_geojson
)
from flood_alert import send_flood_alert

st.set_page_config(
    page_title="Flood Monitor — India",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Flood Detection & Alert System")
st.markdown(
    "Real-time flood monitoring for Indian river basins using "
    "Sentinel-2 satellite imagery. Automatic alerts to NDRF/SDRF teams."
)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown("**📍 Select location**")
    basin = st.selectbox(
        "River basin",
        list(INDIAN_BASINS.keys())
    )

    basin_info = INDIAN_BASINS[basin]

    if basin == "Custom location":
        st.markdown("**Custom bounding box**")
        min_lon = st.number_input("Min longitude", value=77.0)
        min_lat = st.number_input("Min latitude", value=28.0)
        max_lon = st.number_input("Max longitude", value=78.0)
        max_lat = st.number_input("Max latitude", value=29.0)
        bbox = [min_lon, min_lat, max_lon, max_lat]
    else:
        bbox = basin_info["bbox"]
        center_lat = (bbox[1] + bbox[3]) / 2
        center_lon = (bbox[0] + bbox[2]) / 2

    st.markdown("---")
    st.markdown("**🛰️ Copernicus credentials**")
    cop_user = st.text_input("Username")
    cop_pass = st.text_input("Password", type="password")

    st.markdown("---")
    st.markdown("**📧 Email alerts**")
    sender_email = st.text_input("Your Gmail")
    sender_pass = st.text_input("Gmail App Password", type="password")
    recipient = st.text_input(
        "Alert recipients (comma separated)",
        placeholder="ndrf@example.com, collector@example.com"
    )
    send_alerts = st.checkbox("Send email alert if flood detected", value=True)

    st.markdown("---")
    st.markdown("**📅 Date range**")
    date1 = st.date_input("Baseline date (before)",
                           value=datetime.now() - timedelta(days=30))
    date2 = st.date_input("Current date (after)",
                           value=datetime.now() - timedelta(days=5))
    cloud_pct = st.slider("Max cloud cover %", 0, 50, 30)

# ── Basin info ────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📍 {basin}")
    if basin != "Custom location":
        risk_colors = {
            "extreme": "🔴",
            "high": "🟠",
            "moderate": "🟡",
            "low": "🟢",
            "unknown": "⚪"
        }
        risk_icon = risk_colors.get(basin_info["risk"], "⚪")
        st.markdown(f"{risk_icon} **Flood risk:** {basin_info['risk'].upper()}")
        st.markdown(f"📝 {basin_info['description']}")

        # Show on map
        m = folium.Map(
            location=[(bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2],
            zoom_start=7
        )
        folium.Rectangle(
            bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            color="red",
            fill=True,
            fill_opacity=0.2,
            popup=f"{basin} — Monitoring area"
        ).add_to(m)
        st_folium(m, height=300, use_container_width=True)

with col2:
    st.subheader("📊 Quick stats")
    if basin != "Custom location":
        area_km2 = ((bbox[2]-bbox[0]) * 111) * ((bbox[3]-bbox[1]) * 111)
        st.metric("Monitoring area", f"{area_km2:.0f} km²")
        st.metric("Flood risk", basin_info["risk"].upper())
        st.metric("Satellite revisit", "Every 5 days")
        st.metric("Resolution", "10m per pixel")


# ── Upload mode ───────────────────────────────────────────
st.markdown("---")
st.subheader("🛰️ Analyse flood — upload or download satellite data")

mode = st.radio(
    "Data source",
    ["Upload images manually", "Download from Copernicus (live)"],
    horizontal=True
)

if mode == "Upload images manually":
    st.info(
        "Upload Sentinel-2 RGB images (before and after). "
        "You can download these from "
        "[Copernicus Browser](https://browser.dataspace.copernicus.eu/) "
        "or use the t1_rgb.png and t2_rgb.png from your data/sentinel folder."
    )
    c1, c2 = st.columns(2)
    with c1:
        t1_file = st.file_uploader(
            "Before image (baseline)", type=["png", "jpg", "jpeg"])
        if t1_file:
            t1_img = np.array(
                Image.open(t1_file).convert("RGB").resize((512, 512)))
            st.image(t1_img, use_container_width=True)

    with c2:
        t2_file = st.file_uploader(
            "After image (current)", type=["png", "jpg", "jpeg"])
        if t2_file:
            t2_img = np.array(
                Image.open(t2_file).convert("RGB").resize((512, 512)))
            st.image(t2_img, use_container_width=True)

    run_button = st.button(
        "🔍 Detect Flood", type="primary", use_container_width=True)

    if run_button:
        if not t1_file or not t2_file:
            st.error("Please upload both before and after images.")
            st.stop()

        with st.spinner("Analysing flood extent..."):
            metrics = calculate_flood_metrics(t1_img, t2_img)

            os.makedirs("outputs/flood", exist_ok=True)
            map_path = "outputs/flood/flood_map.png"
            generate_flood_map(
                t1_img, t2_img, metrics, map_path, basin)

            geojson_path = "outputs/flood/flood_extent.geojson"
            export_geojson(
                metrics, bbox, geojson_path,
                basin, str(date2))

        # Results
        st.markdown("---")
        severity = metrics["severity"]
        color = metrics["severity_color"]

        st.markdown(
            f"<h2 style='color:{color}'>⚠️ {severity} FLOOD DETECTED</h2>"
            if severity != "NORMAL"
            else "<h2 style='color:#3B6D11'>✅ NO SIGNIFICANT FLOOD</h2>",
            unsafe_allow_html=True
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Severity", severity)
        m2.metric("New flood area", f"{metrics['flooded_km2']} km²")
        m3.metric("Scene affected", f"{metrics['flood_pct']}%")
        m4.metric("Water increase", f"+{metrics['water_increase_pct']}%")

        st.markdown("---")
        st.subheader("🗺️ Flood assessment map")
        st.image(map_path, use_container_width=True)

        # Downloads
        c1, c2 = st.columns(2)
        with c1:
            with open(map_path, "rb") as f:
                st.download_button(
                    "⬇️ Download flood map",
                    f.read(),
                    "flood_map.png",
                    "image/png",
                    use_container_width=True
                )
        with c2:
            with open(geojson_path, "rb") as f:
                st.download_button(
                    "⬇️ Download GeoJSON (Google Earth)",
                    f.read(),
                    "flood_extent.geojson",
                    "application/json",
                    use_container_width=True
                )

        # Send alert
        if (send_alerts and severity in ["CRITICAL", "SEVERE", "MODERATE"]
                and sender_email and sender_pass and recipient):
            with st.spinner("Sending email alert..."):
                recipients = [r.strip() for r in recipient.split(",")]
                success = send_flood_alert(
                    sender_email, sender_pass,
                    recipients, basin, metrics,
                    map_path, geojson_path
                )
                if success:
                    st.success(
                        f"✅ Alert sent to {len(recipients)} recipient(s)")
                else:
                    st.warning(
                        "Alert failed — check Gmail App Password settings")

        elif severity == "NORMAL":
            st.success("No significant flooding detected — no alert sent.")

else:
    st.info(
        "Enter Copernicus credentials in the sidebar, "
        "select dates, then click Download & Analyse."
    )
    if st.button("🛰️ Download & Analyse",
                 type="primary", use_container_width=True):
        if not cop_user or not cop_pass:
            st.error("Enter Copernicus credentials in sidebar.")
            st.stop()

        try:
            from sentinel import get_sentinel_pair
            with st.spinner("Downloading Sentinel-2 data... (5-15 mins)"):
                t1_path, t2_path = get_sentinel_pair(
                    cop_user, cop_pass, bbox,
                    str(date1), str(date2),
                    cloud_pct=cloud_pct
                )
            t1_img = np.array(
                Image.open(t1_path).convert("RGB").resize((512, 512)))
            t2_img = np.array(
                Image.open(t2_path).convert("RGB").resize((512, 512)))
            st.success("Downloaded ✅ — running flood analysis...")
            st.rerun()

        except Exception as e:
            st.error(f"Download failed: {e}")