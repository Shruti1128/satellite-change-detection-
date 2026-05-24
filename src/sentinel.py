import os
import requests
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import json


class CopernicusClient:
    """
    Client for Copernicus Data Space Ecosystem API.
    Downloads real Sentinel-2 L2A imagery for any location and date.
    """

    AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.token = None
        self._authenticate()

    def _authenticate(self):
        """Get access token from Copernicus."""
        response = requests.post(self.AUTH_URL, data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
        })
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            print("Copernicus authentication successful ✅")
        else:
            raise Exception(
                f"Authentication failed: {response.status_code} {response.text}"
            )

    def search_scenes(self, bbox, date_str, days_window=15, cloud_cover=20):
        """
        Search for Sentinel-2 scenes near a date with low cloud cover.

        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            date_str: 'YYYY-MM-DD'
            days_window: search ±days around the date
            max_cloud: maximum cloud cover percentage

        Returns:
            list of scene metadata dicts
        """
        date = datetime.strptime(date_str, "%Y-%m-%d")
        start = (date - timedelta(days=days_window)).strftime("%Y-%m-%dT00:00:00.000Z")
        end   = (date + timedelta(days=days_window)).strftime("%Y-%m-%dT23:59:59.000Z")

        # WKT polygon from bbox
        min_lon, min_lat, max_lon, max_lat = bbox
        wkt = (f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
               f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))")

        params = {
            "$filter": (
                f"Collection/Name eq 'SENTINEL-2' and "
                f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
                f"ContentDate/Start gt {start} and "
                f"ContentDate/Start lt {end} and "
                f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq "
                f"'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {cloud_cover})"
            ),
            "$orderby": "ContentDate/Start asc",
            "$top": 5,
        }

        response = requests.get(self.SEARCH_URL, params=params)
        if response.status_code != 200:
            raise Exception(f"Search failed: {response.status_code}")

        results = response.json().get("value", [])
        print(f"Found {len(results)} scenes for {date_str} ±{days_window} days")
        return results

    def download_scene(self, scene_id, scene_name, output_dir):
        """Download and extract a Sentinel-2 scene."""
        os.makedirs(output_dir, exist_ok=True)
        zip_path = os.path.join(output_dir, f"{scene_name}.zip")

        headers = {"Authorization": f"Bearer {self.token}"}
        url = f"{self.DOWNLOAD_URL}({scene_id})/$value"

        print(f"Downloading {scene_name}...")
        response = requests.get(url, headers=headers, stream=True)

        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}%", end="", flush=True)

        print(f"\nDownloaded to {zip_path}")

        # Extract
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(output_dir)
        os.remove(zip_path)

        return os.path.join(output_dir, scene_name + ".SAFE")


def find_band_file(safe_dir, band="B04", resolution="10m"):
    """Find a specific band file inside a .SAFE directory."""
    for root, dirs, files in os.walk(safe_dir):
        for f in files:
            if f.endswith(".jp2"):
                if band in f:
                    # For L2A with resolution
                    if resolution and resolution in f:
                        return os.path.join(root, f)
                    # For L1C without resolution in filename
                    elif not resolution:
                        return os.path.join(root, f)
    # Second pass — return any match with band name
    for root, dirs, files in os.walk(safe_dir):
        for f in files:
            if f.endswith(".jp2") and band in f:
                return os.path.join(root, f)
    return None

def safe_to_rgb(safe_dir, output_path, size=1024):
    try:
        import rasterio
        from rasterio.enums import Resampling

        # Try L2A format first (10m resolution suffix)
        b04 = find_band_file(safe_dir, "B04", "10m")
        b03 = find_band_file(safe_dir, "B03", "10m")
        b02 = find_band_file(safe_dir, "B02", "10m")

        # Fall back to L1C format (no resolution suffix)
        if not b04:
            b04 = find_band_file(safe_dir, "B04", "")
        if not b03:
            b03 = find_band_file(safe_dir, "B03", "")
        if not b02:
            b02 = find_band_file(safe_dir, "B02", "")

        print(f"B04: {b04}")
        print(f"B03: {b03}")
        print(f"B02: {b02}")

        if not all([b04, b03, b02]):
            raise Exception(f"Could not find RGB bands in {safe_dir}")

        bands = []
        for band_path in [b04, b03, b02]:
            with rasterio.open(band_path) as src:
                data = src.read(
                    1,
                    out_shape=(size, size),
                    resampling=Resampling.bilinear
                ).astype(np.float32)
            bands.append(data)

        rgb = np.stack(bands, axis=-1)
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)

        Image.fromarray(rgb).save(output_path)
        print(f"Saved RGB image: {output_path}")
        return output_path

    except ImportError:
        print("rasterio not installed")
        return None


def get_sentinel_pair(username, password, bbox, date1, date2,
                      output_dir="data/sentinel", cloud_pct=20):
    """
    Full pipeline: authenticate → search → download → convert to RGB PNG pair.

    Args:
        username, password: Copernicus credentials
        bbox: [min_lon, min_lat, max_lon, max_lat]
        date1: 'YYYY-MM-DD' before date
        date2: 'YYYY-MM-DD' after date
        output_dir: where to save files
        cloud_pct: max cloud cover %

    Returns:
        (t1_path, t2_path) — paths to RGB PNG images ready for model
    """
    client = CopernicusClient(username, password)

    os.makedirs(output_dir, exist_ok=True)

    # Search for scenes
    scenes1 = client.search_scenes(bbox, date1, cloud_cover=cloud_pct)
    scenes2 = client.search_scenes(bbox, date2, cloud_cover=cloud_pct)

    if not scenes1:
        raise Exception(f"No scenes found for date {date1} with cloud < {cloud_pct}%")
    if not scenes2:
        raise Exception(f"No scenes found for date {date2} with cloud < {cloud_pct}%")

    scene1 = scenes1[0]
    scene2 = scenes2[0]

    print(f"T1 scene: {scene1['Name']}")
    print(f"T2 scene: {scene2['Name']}")

    # Download
    safe1 = client.download_scene(scene1["Id"], scene1["Name"], output_dir)
    safe2 = client.download_scene(scene2["Id"], scene2["Name"], output_dir)

    # Convert to RGB
    t1_path = os.path.join(output_dir, "t1_rgb.png")
    t2_path = os.path.join(output_dir, "t2_rgb.png")

    safe_to_rgb(safe1, t1_path)
    safe_to_rgb(safe2, t2_path)

    return t1_path, t2_path


if __name__ == "__main__":
    # Quick test — search for scenes over Delhi
    import os

    username = os.environ.get("COPERNICUS_USER", "your_username")
    password = os.environ.get("COPERNICUS_PASS", "your_password")

    # Delhi bounding box
    bbox = [76.8, 28.4, 77.4, 28.9]

    client = CopernicusClient(username, password)
    scenes = client.search_scenes(bbox, "2024-01-15")

    if scenes:
        print(f"\nFirst scene: {scenes[0]['Name']}")
        print(f"Date: {scenes[0]['ContentDate']['Start']}")
    else:
        print("No scenes found")