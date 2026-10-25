"""Composite rendered cameras and validate RGB corner-training samples."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from predict_corner.data_prep.generate_corner_heatmaps import (
    make_corner_heatmaps, check_heatmaps, draw_alignment, draw_channels,
)


def procedural_background(rng, size):
    width, height = size
    style = str(rng.choice(["grid_table", "denim", "wood", "wall"]))
    ys, xs = np.mgrid[:height, :width]
    noise = rng.normal(0, 2, (height, width, 1))
    if style == "denim":
        color = np.array([47, 68, 91]) * rng.uniform(.7, 1.5)
        pattern = 5 * np.sin((xs + ys) * 2.2) + 2 * np.sin(xs * 1.7)
    elif style == "wood":
        color = np.array([136, 107, 78]) * rng.uniform(.7, 1.4)
        pattern = 8 * np.sin(xs * .07 + 2 * np.sin(ys * .014))
    else:
        color = np.repeat(rng.uniform(125, 220), 3)
        pattern = 7 * np.sin(xs * .005 + ys * .004)
    pixels = color + pattern[..., None] + noise
    image = Image.fromarray(pixels.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    if style == "grid_table":
        spacing = int(rng.integers(25, 60))
        shift = int(rng.integers(-15, 16))
        for x in range(-100, width + 100, spacing):
            draw.line((x, 0, x + shift * 3, height), fill=(65, 70, 72), width=2)
        for y in range(-spacing, height + spacing, spacing):
            draw.line((0, y, width, y + shift), fill=(65, 70, 72), width=2)
        if rng.random() < .65:
            x0, y0 = int(rng.integers(-100, 250)), int(rng.integers(-100, 200))
            draw.rounded_rectangle((x0, y0, x0 + 400, y0 + 450), radius=25, fill=(49, 66, 88))
            draw.line((x0 + 210, y0, x0 + 210, y0 + 450), fill=(110, 122, 139), width=3)
            for y in range(y0 + 25, y0 + 450, 65):
                draw.ellipse((x0 + 202, y, x0 + 216, y + 14), fill=(160, 161, 160))
    return image, "procedural:" + style


def make_background(rng, size, sources):
    if sources and rng.random() < .45:
        path = sources[int(rng.integers(len(sources)))]
        with Image.open(path) as image:
            image = image.convert("RGB")
            fraction = float(rng.uniform(.45, 1.0))
            crop_width, crop_height = int(image.width * fraction), int(image.height * fraction)
            x = int(rng.integers(image.width - crop_width + 1))
            y = int(rng.integers(image.height - crop_height + 1))
            image = image.crop((x, y, x + crop_width, y + crop_height)).resize(size, Image.Resampling.BILINEAR)
        if rng.random() < .5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image, str(path)
    return procedural_background(rng, size)


def occluder(rng, mask):
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    extent = max(np.ptp(xs), np.ptp(ys))
    for _ in range(100):
        layer = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(layer)
        angle = float(rng.uniform(0, 2 * np.pi))
        center = np.array([xs.mean(), ys.mean()]) + rng.uniform(-.6, .6, 2) * extent
        delta = np.array([np.cos(angle), np.sin(angle)]) * extent * rng.uniform(.6, 1.8)
        thickness = int(rng.uniform(.12, .45) * extent)
        a, b = center - delta, center + delta
        draw.line((tuple(a), tuple(b)), fill=255, width=thickness)
        radius = thickness / 2
        for x, y in (a, b):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        pixels = np.asarray(layer) > 127
        fraction = float((pixels & mask).sum() / mask.sum())
        if .05 <= fraction <= .4:
            kind = str(rng.choice(["skin", "strap", "cloth"]))
            palettes = {"skin": [185, 139, 109], "strap": [46, 49, 48], "cloth": [49, 69, 92]}
            color = np.array(palettes[kind]) * rng.uniform(.7, 1.3)
            shade = np.linspace(.8, 1.1, width)[None, :, None]
            texture = color * shade + rng.normal(0, 2, (height, width, 1))
            image = Image.fromarray(texture.clip(0, 255).astype(np.uint8))
            return layer, image, kind, fraction
    raise RuntimeError("Could not sample a partial occluder")


def finish(scene, background_paths, store_heatmaps=False):
    manifest = json.loads((scene / "render_manifest.json").read_text())
    assert manifest["status"] == "rendered"
    labels = json.loads((scene / "keypoints_2d.json").read_text())
    poses = json.loads((scene / "scene_gt.json").read_text())
    cameras = json.loads((scene / "scene_camera.json").read_text())
    corners_3d = np.array(json.loads((scene / "keypoints_3d.json").read_text())["corners"])
    for folder in ("rgb", "mask", "mask_visib", "checks"):
        (scene / folder).mkdir(exist_ok=True)
    if store_heatmaps:
        (scene / "heatmaps_boxdreamer").mkdir(exist_ok=True)
    previews, records = [], []
    projection_errors, peak_errors = [], []
    for index in range(manifest["count"]):
        rng = np.random.default_rng(manifest["seed"] + index * 1009 + 7919)
        stem, key = f"{index:06d}", str(index)
        corners = np.array(labels["images"][key]["corners_2d"])
        pose = poses[key][0]
        R = np.array(pose["cam_R_m2c"]).reshape(3, 3)
        t = np.array(pose["cam_t_m2c"])
        K = np.array(cameras[key]["cam_K"]).reshape(3, 3)
        projected, _ = cv2.projectPoints(corners_3d, cv2.Rodrigues(R)[0], t, K, None)
        error = float(np.abs(projected.reshape(8, 2) - corners).max())
        assert error < .001
        projection_errors.append(error)
        with Image.open(scene / "rgba" / (stem + ".png")) as source:
            rgba = source.convert("RGBA")
        mask = np.asarray(rgba)[..., 3] >= 128
        assert mask.sum() > 300
        ys, xs = np.nonzero(mask)
        assert xs.min() >= corners[:, 0].min() - 2 and xs.max() <= corners[:, 0].max() + 2
        assert ys.min() >= corners[:, 1].min() - 2 and ys.max() <= corners[:, 1].max() + 2
        background, background_source = make_background(rng, rgba.size, background_paths)
        rgb = Image.alpha_composite(background.convert("RGBA"), rgba).convert("RGB")
        visible = mask.copy()
        kind, fraction = "none", 0.
        if index % 5 in (1, 2, 4):
            cover, texture, kind, fraction = occluder(rng, mask)
            rgb = Image.composite(texture, rgb, cover)
            visible &= np.asarray(cover) < 128
        brightness, contrast = float(rng.uniform(.75, 1.25)), float(rng.uniform(.8, 1.2))
        rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
        rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
        exposure_ev = float(rng.uniform(-.65, .55))
        gamma = float(rng.uniform(.85, 1.15))
        white_balance = str(rng.choice(["warm", "neutral", "cool"]))
        balances = {"warm": [1.12, 1., .86], "neutral": [1., 1., 1.], "cool": [.86, 1., 1.12]}
        values = np.asarray(rgb, dtype=np.float32) / 255
        values = np.clip(values * (2 ** exposure_ev) * balances[white_balance], 0, 1) ** gamma
        pixels = (values * 255).clip(0, 255).astype(np.uint8)
        blur = "none"
        if index % 5 == 2:
            sigma = float(rng.uniform(.4, 1.3))
            pixels = cv2.GaussianBlur(pixels, (5, 5), sigma)
            blur = f"gaussian:{sigma:.3f}"
        elif index % 5 == 3:
            length = int(rng.choice([3, 5, 7, 9]))
            kernel = np.zeros((length, length), np.float32)
            kernel[length // 2] = 1 / length
            if rng.random() < .5:
                kernel = kernel.T
            pixels = cv2.filter2D(pixels, -1, kernel)
            blur = f"motion:{length}"
        sigma = float(rng.uniform(0, 5))
        pixels = np.clip(pixels.astype(np.float32) + rng.normal(0, sigma, pixels.shape), 0, 255).astype(np.uint8)
        rgb = Image.fromarray(pixels)
        rgb.save(scene / "rgb" / (stem + ".png"))
        Image.fromarray(mask.astype(np.uint8) * 255).save(scene / "mask" / (stem + "_000000.png"))
        Image.fromarray(visible.astype(np.uint8) * 255).save(scene / "mask_visib" / (stem + "_000000.png"))
        heatmaps, center, denominators = make_corner_heatmaps(corners, rgba.size)
        check = check_heatmaps(heatmaps, corners, center, denominators)
        peak_errors.extend(check["peak_errors_pixels"])
        assert all(check["in_image"])
        if store_heatmaps:
            np.savez_compressed(scene / "heatmaps_boxdreamer" / (stem + ".npz"), heatmaps=heatmaps,
                                corners_xy=corners, center_xy=center, denominators=denominators,
                                in_image=np.ones(8, dtype=bool), image_size_wh=np.array(rgba.size))
        if index < 24 or index % 50 == 0:
            preview = draw_alignment(rgb, corners, np.array(check["peak_xy"]), labels["edges"])
            preview.save(scene / "checks" / (stem + "_corners.jpg"), quality=94)
            previews.append((index, preview, kind, blur))
        if index == 0:
            draw_channels(rgb, heatmaps, corners, np.array(check["peak_xy"])).save(scene / "checks/eight_channels.jpg", quality=94)
        records.append({"image_id": index, "background": background_source, "occluder": kind,
                        "occluded_fraction": fraction, "blur": blur, "noise_sigma": sigma,
                        "brightness": brightness, "contrast": contrast,
                        "exposure_ev": exposure_ev, "gamma": gamma, "white_balance": white_balance,
                        "projection_error_px": error, "rgb_sha256": hashlib.sha256((scene / "rgb" / (stem + ".png")).read_bytes()).hexdigest()})
        if (index + 1) % 50 == 0 or index + 1 == manifest["count"]:
            print(f"COMPOSITED {index + 1}/{manifest['count']}", flush=True)
    for start in range(0, len(previews), 24):
        sheet = Image.new("RGB", (1600, 5 * 268), "white")
        for slot, (index, image, kind, blur) in enumerate(previews[start:start + 24]):
            x, y = slot % 5 * 320, slot // 5 * 268
            sheet.paste(image.resize((320, 240)), (x, y))
            ImageDraw.Draw(sheet).text((x + 5, y + 243), f"{index:06d} / {kind} / {blur}", fill="black")
        sheet.save(scene / "checks" / f"preview_{start // 24:02d}.jpg", quality=94)
    report = {"status": "PASS", "count": len(records), "seed": manifest["seed"],
              "occluded_images": sum(r["occluder"] != "none" for r in records),
              "blurred_images": sum(r["blur"] != "none" for r in records),
              "max_projection_error_px": max(projection_errors), "max_heatmap_peak_error_px": max(peak_errors),
              "heatmaps": "stored float32 NPZ" if store_heatmaps else "generated during training using the unchanged paper formula",
              "backgrounds": "procedural grid/cloth/wood/wall plus independent HCCEPose example photos",
              "foreground_occlusion": "procedural skin-colored, strap-like and cloth-like masks; labels retain all eight corners",
              "source_video_used": False, "frames": records}
    (scene / "augmentation_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2), flush=True)
    return report


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "pilot"], default="train")
    parser.add_argument("--store-heatmaps", action="store_true")
    args = parser.parse_args()
    files = sorted((root / "HCCEPose/test_imgs").glob("*.jpg")) + sorted((root / "HCCEPose/test_imgs_RGBD").glob("*_rgb.png"))
    sources = files[:3] if args.split in ("train", "pilot") else files[3:]
    finish(args.scene.resolve(), sources, args.store_heatmaps)


if __name__ == "__main__":
    main()
