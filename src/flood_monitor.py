import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Known flood-prone river basins in India
INDIAN_BASINS = {
    "Brahmaputra — Assam": {
        "bbox": [90.5, 25.5, 92.5, 27.5],
        "description": "Highly flood-prone, affects millions annually",
        "risk": "extreme"
    },
    "Ganga — Bihar/UP": {
        "bbox": [84.0, 25.0, 86.0, 27.0],
        "description": "Major flooding during monsoon season",
        "risk": "high"
    },
    "Yamuna — Delhi": {
        "bbox": [76.8, 28.2, 77.6, 29.0],
        "description": "Urban flooding risk for Delhi",
        "risk": "high"
    },
    "Godavari — Andhra": {
        "bbox": [81.0, 16.5, 82.5, 18.0],
        "description": "Severe flooding in delta region",
        "risk": "high"
    },
    "Mahanadi — Odisha": {
        "bbox": [83.5, 19.5, 85.5, 21.0],
        "description": "Cyclone-induced flooding",
        "risk": "extreme"
    },
    "Kosi — Bihar": {
        "bbox": [86.0, 25.5, 87.5, 27.0],
        "description": "Known as Sorrow of Bihar",
        "risk": "extreme"
    },
    "Custom location": {
        "bbox": None,
        "description": "Enter your own coordinates",
        "risk": "unknown"
    }
}


def detect_water_pixels(image_arr, threshold=0.15):
    """
    Detect water pixels using NDWI-like index on RGB.
    Water appears blue-ish and dark in RGB satellite imagery.

    Uses: water = (B - R) / (B + R + epsilon)
    High values = water, Low values = land

    Args:
        image_arr: np.ndarray [H, W, 3] RGB uint8
        threshold: detection threshold

    Returns:
        water_mask: np.ndarray [H, W] bool
        ndwi: np.ndarray [H, W] float32
    """
    img = image_arr.astype(np.float32) / 255.0
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # Modified NDWI using RGB
    # Water is typically: low R, low G, relatively higher B
    # Also dark overall (low brightness)
    ndwi = (G - R) / (G + R + 1e-6)
    darkness = 1.0 - (R + G + B) / 3.0

    # Combined water index
    water_index = 0.6 * ndwi + 0.4 * darkness
    water_mask = water_index > threshold

    return water_mask, water_index


def calculate_flood_metrics(t1_arr, t2_arr,
                             pixel_size_m=10.0,
                             threshold=0.15):
    """
    Compare before/after images to detect flood extent.

    Args:
        t1_arr: before image [H, W, 3] uint8
        t2_arr: after image [H, W, 3] uint8
        pixel_size_m: pixel size in metres (10m for Sentinel-2)
        threshold: water detection threshold

    Returns:
        dict with flood metrics
    """
    H, W = t1_arr.shape[:2]
    total_pixels = H * W
    pixel_area_km2 = (pixel_size_m / 1000) ** 2

    # Detect water in both images
    water_t1, index_t1 = detect_water_pixels(t1_arr, threshold)
    water_t2, index_t2 = detect_water_pixels(t2_arr, threshold)

    # New flood = was land before, is water now
    new_flood = (~water_t1) & water_t2

    # Receded water = was water before, is land now
    receded = water_t1 & (~water_t2)

    # Persistent water = water in both
    persistent = water_t1 & water_t2

    # Calculate areas
    baseline_water_km2 = water_t1.sum() * pixel_area_km2
    current_water_km2 = water_t2.sum() * pixel_area_km2
    flooded_km2 = new_flood.sum() * pixel_area_km2
    receded_km2 = receded.sum() * pixel_area_km2

    # Flood severity
    flood_pct = (new_flood.sum() / total_pixels) * 100
    if flood_pct > 20:
        severity = "CRITICAL"
        severity_color = "#E24B4A"
    elif flood_pct > 10:
        severity = "SEVERE"
        severity_color = "#EF9F27"
    elif flood_pct > 3:
        severity = "MODERATE"
        severity_color = "#F5C842"
    elif flood_pct > 0.5:
        severity = "MINOR"
        severity_color = "#378ADD"
    else:
        severity = "NORMAL"
        severity_color = "#3B6D11"

    return {
        "severity": severity,
        "severity_color": severity_color,
        "baseline_water_km2": round(baseline_water_km2, 2),
        "current_water_km2": round(current_water_km2, 2),
        "flooded_km2": round(flooded_km2, 2),
        "receded_km2": round(receded_km2, 2),
        "flood_pct": round(flood_pct, 2),
        "water_increase_pct": round(
            (current_water_km2 - baseline_water_km2) /
            (baseline_water_km2 + 1e-6) * 100, 1
        ),
        "new_flood_mask": new_flood,
        "water_t1": water_t1,
        "water_t2": water_t2,
        "water_index_t1": index_t1,
        "water_index_t2": index_t2,
        "total_area_km2": round(total_pixels * pixel_area_km2, 2),
    }


def generate_flood_map(t1_arr, t2_arr, metrics,
                        output_path, location_name=""):
    """
    Generate a 4-panel flood assessment visualisation.
    Panel 1: Before image
    Panel 2: After image
    Panel 3: Water extent map
    Panel 4: Flood damage map (new inundation highlighted)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#0F1117")

    title = f"Flood Assessment — {location_name}" if location_name else "Flood Assessment"
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=1.02)

    # Panel 1 — Before
    axes[0].imshow(t1_arr)
    axes[0].set_title("Before", color="white", fontsize=11)
    axes[0].axis("off")

    # Panel 2 — After
    axes[1].imshow(t2_arr)
    axes[1].set_title("After", color="white", fontsize=11)
    axes[1].axis("off")

    # Panel 3 — Water extent
    water_map = np.zeros((*t2_arr.shape[:2], 3), dtype=np.uint8)
    water_map[metrics["water_t1"]] = [100, 149, 237]   # blue — baseline water
    water_map[metrics["water_t2"]] = [0, 100, 255]     # bright blue — current water
    axes[2].imshow(water_map)
    axes[2].set_title("Water extent", color="white", fontsize=11)
    axes[2].axis("off")

    # Panel 4 — Flood damage
    damage_map = t2_arr.copy()
    # Highlight new flood in red
    damage_map[metrics["new_flood_mask"], 0] = 255
    damage_map[metrics["new_flood_mask"], 1] = 50
    damage_map[metrics["new_flood_mask"], 2] = 50
    axes[3].imshow(damage_map)
    axes[3].set_title(
        f"New flood — {metrics['flooded_km2']} km²",
        color="#E24B4A", fontsize=11
    )
    axes[3].axis("off")

    # Style
    for ax in axes:
        ax.set_facecolor("#0F1117")

    # Metrics text
    metrics_text = (
        f"Severity: {metrics['severity']}  |  "
        f"Flooded: {metrics['flooded_km2']} km²  |  "
        f"Water increase: {metrics['water_increase_pct']}%  |  "
        f"Affected area: {metrics['flood_pct']}% of scene"
    )
    fig.text(0.5, -0.02, metrics_text,
             ha="center", color="white", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="#1E2130", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150,
                bbox_inches="tight",
                facecolor="#0F1117")
    plt.close()
    print(f"Flood map saved: {output_path}")
    return output_path


def export_geojson(metrics, bbox, output_path, location_name="", date_str=""):
    """
    Export flood detection results as GeoJSON.
    Can be opened in Google Earth, QGIS, or ArcGIS.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat

    H, W = metrics["new_flood_mask"].shape
    flood_mask = metrics["new_flood_mask"]

    # Convert flood pixels to geographic coordinates
    features = []

    # Add bounding box as overall extent
    features.append({
        "type": "Feature",
        "properties": {
            "name": f"Flood extent — {location_name}",
            "date": date_str,
            "severity": metrics["severity"],
            "flooded_km2": metrics["flooded_km2"],
            "flood_pct": metrics["flood_pct"],
            "water_increase_pct": metrics["water_increase_pct"],
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat]
            ]]
        }
    })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "generated": datetime.now().isoformat(),
            "location": location_name,
            "severity": metrics["severity"],
            "flooded_area_km2": metrics["flooded_km2"],
            "total_scene_km2": metrics["total_area_km2"],
        }
    }

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"GeoJSON saved: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Flood monitor module loaded successfully")
    print(f"Monitoring {len(INDIAN_BASINS)} Indian river basins")
    for name, info in INDIAN_BASINS.items():
        print(f"  {name} — Risk: {info['risk']}")