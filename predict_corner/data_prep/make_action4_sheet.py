"""Create a diagnostic contact sheet with the fixed 8-corner ordering."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=Path, required=True)
args = parser.parse_args()
checks = args.dataset / "checks"
report = json.loads((checks / "projection_check.json").read_text())
sheet = Image.new("RGB", (1280, 1024), "#eeeeee")
for i, view in enumerate(report["views"]):
    with Image.open(checks / view["image"]) as source:
        preview = source.convert("RGB")
    draw = ImageDraw.Draw(preview)
    points = [tuple(point) for point in view["corners_2d"]]
    for edge in report["edges"]:
        draw.line((points[edge[0]], points[edge[1]]), fill="#ef4444", width=2)
    for j, (x, y) in enumerate(points):
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#ffd500", outline="black")
        draw.text((x + 5, y + 2), str(j), fill="black", stroke_width=1, stroke_fill="white")
    preview.save(checks / (Path(view["image"]).stem + "_bbox.png"))
    x, y = i % 2 * 640, i // 2 * 512
    sheet.paste(preview, (x, y))
    ImageDraw.Draw(sheet).text((x + 10, y + 486), view["name"] + " / BOP model + projected cuboid", fill="black")
sheet.save(checks / "preparation_contact_sheet.jpg", quality=94)
print(checks / "preparation_contact_sheet.jpg")
