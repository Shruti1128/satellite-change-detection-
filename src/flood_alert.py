import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime


def send_flood_alert(
    sender_email,
    sender_password,
    recipient_emails,
    location_name,
    metrics,
    image_path=None,
    geojson_path=None
):
    """
    Send flood alert email with assessment image attached.

    Args:
        sender_email: Gmail address to send from
        sender_password: Gmail app password
        recipient_emails: list of recipient emails
        location_name: name of monitored location
        metrics: flood metrics dict from flood_monitor
        image_path: path to flood map image
        geojson_path: path to GeoJSON file
    """
    severity = metrics["severity"]
    flooded_km2 = metrics["flooded_km2"]
    flood_pct = metrics["flood_pct"]
    water_increase = metrics["water_increase_pct"]

    # Subject line based on severity
    emoji = {
        "CRITICAL": "🚨",
        "SEVERE": "⚠️",
        "MODERATE": "🔶",
        "MINOR": "🔵",
        "NORMAL": "✅"
    }.get(severity, "⚠️")

    subject = f"{emoji} FLOOD ALERT [{severity}] — {location_name} — {datetime.now().strftime('%d %b %Y')}"

    # HTML email body
    color = {
        "CRITICAL": "#E24B4A",
        "SEVERE": "#EF9F27",
        "MODERATE": "#F5C842",
        "MINOR": "#378ADD",
        "NORMAL": "#3B6D11"
    }.get(severity, "#EF9F27")

    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px;">

    <div style="max-width: 600px; margin: 0 auto; background: white;
                border-radius: 12px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">

        <div style="background: {color}; padding: 24px; text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 24px;">
                {emoji} FLOOD ALERT
            </h1>
            <p style="color: white; margin: 8px 0 0; font-size: 16px; opacity: 0.9;">
                Severity: <strong>{severity}</strong>
            </p>
        </div>

        <div style="padding: 24px;">
            <h2 style="color: #333; margin-top: 0;">{location_name}</h2>
            <p style="color: #666;">
                Detected: {datetime.now().strftime('%d %B %Y at %H:%M IST')}
            </p>

            <table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
                <tr style="background: #f9f9f9;">
                    <td style="padding: 12px; border: 1px solid #eee; font-weight: bold;">
                        New flooded area
                    </td>
                    <td style="padding: 12px; border: 1px solid #eee; color: {color}; font-size: 18px;">
                        <strong>{flooded_km2} km²</strong>
                    </td>
                </tr>
                <tr>
                    <td style="padding: 12px; border: 1px solid #eee; font-weight: bold;">
                        Scene affected
                    </td>
                    <td style="padding: 12px; border: 1px solid #eee;">
                        {flood_pct}% of monitored area
                    </td>
                </tr>
                <tr style="background: #f9f9f9;">
                    <td style="padding: 12px; border: 1px solid #eee; font-weight: bold;">
                        Water body increase
                    </td>
                    <td style="padding: 12px; border: 1px solid #eee;">
                        +{water_increase}% compared to baseline
                    </td>
                </tr>
                <tr>
                    <td style="padding: 12px; border: 1px solid #eee; font-weight: bold;">
                        Baseline water area
                    </td>
                    <td style="padding: 12px; border: 1px solid #eee;">
                        {metrics['baseline_water_km2']} km²
                    </td>
                </tr>
                <tr style="background: #f9f9f9;">
                    <td style="padding: 12px; border: 1px solid #eee; font-weight: bold;">
                        Current water area
                    </td>
                    <td style="padding: 12px; border: 1px solid #eee;">
                        {metrics['current_water_km2']} km²
                    </td>
                </tr>
            </table>

            {'<p style="color: #E24B4A; font-weight: bold;">⚠️ IMMEDIATE ACTION REQUIRED: Deploy NDRF teams to affected area.</p>' if severity == 'CRITICAL' else ''}
            {'<p style="color: #EF9F27; font-weight: bold;">⚠️ Alert district authorities and prepare evacuation routes.</p>' if severity == 'SEVERE' else ''}

            <p style="color: #666; font-size: 13px; margin-top: 24px; border-top: 1px solid #eee; padding-top: 16px;">
                This alert was generated automatically by the Satellite Flood Detection System
                using Sentinel-2 imagery from ESA Copernicus.
                GeoJSON damage map is attached — open in Google Earth for field operations.
            </p>
        </div>
    </div>

    </body>
    </html>
    """

    # Build email
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipient_emails)

    msg.attach(MIMEText(html, "html"))

    # Attach flood map image
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header(
                "Content-Disposition",
                "attachment",
                filename="flood_assessment.png"
            )
            msg.attach(img)

    # Attach GeoJSON
    if geojson_path and os.path.exists(geojson_path):
        with open(geojson_path, "rb") as f:
            geojson_attachment = MIMEText(f.read().decode(), "plain")
            geojson_attachment.add_header(
                "Content-Disposition",
                "attachment",
                filename="flood_extent.geojson"
            )
            msg.attach(geojson_attachment)

    # Send via Gmail SMTP
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(
                sender_email,
                recipient_emails,
                msg.as_string()
            )
        print(f"Alert sent to: {', '.join(recipient_emails)}")
        return True
    except Exception as e:
        print(f"Failed to send alert: {e}")
        print("Make sure you're using a Gmail App Password, not your regular password")
        print("Get one at: myaccount.google.com → Security → App passwords")
        return False


if __name__ == "__main__":
    print("Flood alert module loaded")
    print("Requires Gmail App Password — not your regular Gmail password")
    print("Get one at: myaccount.google.com → Security → App passwords")