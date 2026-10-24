"""Finish and independently validate the 20-frame BOP smoke-test scene."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_png(path, array):
    if path.exists():
        with Image.open(path) as existing:
            if not np.array_equal(np.array(existing), array):
                raise RuntimeError(f"Refusing to overwrite different pixels: {path}")
    else:
        Image.fromarray(array).save(path)


def thumbnail_sheet(images, labels, output):
    sheet = Image.new("RGB", (1600, 4 * 264), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for i, (image, label) in enumerate(zip(images, labels)):
        x, y = i % 5 * 320, i // 5 * 264
        sheet.paste(image.resize((320, 240), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 6, y + 243), label, fill="black")
    sheet.save(output, quality=94)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()
    scene = args.scene.resolve()
    dataset = scene.parent.parent
    manifest_path = scene / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] in ("rendered_pending_postprocess", "verified_smoke_test", "validation_failed")
    assert manifest["count"] == len(manifest["frames"]) == 20
    cameras = json.loads((scene / "scene_camera.json").read_text())
    poses = json.loads((scene / "scene_gt.json").read_text())
    keypoints = json.loads((scene / "keypoints_2d.json").read_text())
    corners = np.array(json.loads((dataset / "keypoints_3d.json").read_text())["corners"], dtype=np.float64)
    assert set(cameras) == set(poses) == set(keypoints["images"]) == {str(i) for i in range(20)}
    info, reports, annotated_images, rgb_images, depth_images, titles = {}, [], [], [], [], []
    alignment_images = []
    expected_names = {f"{i:06d}.png" for i in range(20)}
    for index, frame_info in enumerate(manifest["frames"]):
        stem, key = f"{index:06d}", str(index)
        assert frame_info["image_id"] == index
        depth_m = np.load(scene / "checks/depth_m" / (stem + ".npy"))
        assert depth_m.shape == (480, 640) and np.isfinite(depth_m).all()
        scale = cameras[key]["depth_scale"]
        encoded_float = np.rint(depth_m * 1000 / scale)
        assert encoded_float.max() < 65536 and encoded_float.min() >= 0
        encoded = encoded_float.astype(np.uint16)
        mask = depth_m > 0
        assert np.array_equal(mask, encoded > 0)
        assert not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        mask_u8 = mask.astype(np.uint8) * 255
        save_png(scene / "depth" / (stem + ".png"), encoded)
        for folder in ("mask", "mask_visib"):
            save_png(scene / folder / (stem + "_000000.png"), mask_u8)
        with Image.open(scene / "depth" / (stem + ".png")) as decoded:
            assert np.array_equal(np.array(decoded), encoded)
        with Image.open(scene / "checks/rgba" / (stem + ".png")) as raw:
            rgba = raw.convert("RGBA")
        rgba_array = np.array(rgba)
        assert rgba_array.shape == (480, 640, 4)
        alpha_mask = rgba_array[..., 3] >= 128
        intersection = np.logical_and(mask, alpha_mask).sum()
        union = np.logical_or(mask, alpha_mask).sum()
        iou = float(intersection / union)
        bg = Image.new("RGBA", rgba.size, tuple(frame_info["background_rgb_u8"]) + (255,))
        rgb = Image.alpha_composite(bg, rgba).convert("RGB")
        save_png(scene / "rgb" / (stem + ".png"), np.array(rgb))
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        pixels = int(mask.sum())
        info[key] = [{"bbox_obj": bbox, "bbox_visib": bbox, "px_count_all": pixels,
                      "px_count_valid": pixels, "px_count_visib": pixels, "visib_fract": 1.0}]
        pose = poses[key][0]
        assert pose["obj_id"] == 1
        K = np.array(cameras[key]["cam_K"]).reshape(3, 3)
        R = np.array(pose["cam_R_m2c"]).reshape(3, 3)
        t = np.array(pose["cam_t_m2c"])
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6) and abs(np.linalg.det(R) - 1) < 1e-6
        camera_points = corners @ R.T + t
        assert camera_points[:, 2].min() > 0
        q = camera_points @ K.T
        calculated = q[:, :2] / q[:, 2:]
        labelled = np.array(keypoints["images"][key]["corners_2d"], dtype=np.float64)
        assert labelled.shape == (8, 2) and np.max(np.abs(labelled - calculated)) < 1e-7
        assert np.all((labelled[:, 0] >= 0) & (labelled[:, 0] < 640)
                      & (labelled[:, 1] >= 0) & (labelled[:, 1] < 480))
        projected_lower, projected_upper = labelled.min(axis=0), labelled.max(axis=0)
        assert xs.min() >= projected_lower[0] - 2 and xs.max() <= projected_upper[0] + 2
        assert ys.min() >= projected_lower[1] - 2 and ys.max() <= projected_upper[1] + 2
        success, rvec, tvec = cv2.solvePnP(corners, labelled, K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
        assert success
        recovered_R, _ = cv2.Rodrigues(rvec)
        # Orthogonalize the float32-derived Blender reference only for angular comparison.
        u, _, vh = np.linalg.svd(R)
        exact_R = u @ vh
        cos_angle = np.clip((np.trace(recovered_R @ exact_R.T) - 1) / 2, -1, 1)
        angle_error = float(np.degrees(np.arccos(cos_angle)))
        translation_error = float(np.linalg.norm(tvec.ravel() - t))
        reprojection, _ = cv2.projectPoints(corners, rvec, tvec, K, np.zeros(5))
        pnp_pixel_error = float(np.max(np.linalg.norm(reprojection.reshape(-1, 2) - labelled, axis=1)))
        assert angle_error < .01 and translation_error < .005 and pnp_pixel_error < .001
        image = rgb.copy()
        draw = ImageDraw.Draw(image)
        points = [tuple(point) for point in labelled]
        for edge in keypoints["edges"]:
            draw.line((points[edge[0]], points[edge[1]]), fill="#ef4444", width=2)
        for corner_id, (x, y) in enumerate(points):
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#ffeb00", outline="black")
            draw.text((x + 4, y + 3), str(corner_id), fill="black", stroke_width=1, stroke_fill="white")
        image.save(scene / "checks/overlay" / (stem + ".png"))
        annotated_images.append(image)
        rgb_images.append(rgb)
        depth_vis = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
        a, b = float(depth_m[mask].min()), float(depth_m[mask].max())
        shades = ((depth_m[mask] - a) / max(b - a, 1e-8) * 180 + 50).astype(np.uint8)
        depth_vis[mask] = shades[:, None]
        depth_images.append(Image.fromarray(depth_vis))
        titles.append(f"{stem} / az {frame_info['azimuth_degrees']:.0f} / el {frame_info['elevation_degrees']:.0f}")
        alignment = np.array(rgb).copy()
        for binary, color in ((alpha_mask, (0, 220, 255)), (mask, (0, 255, 30))):
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(alignment, contours, -1, color, 1)
        alignment_images.append(Image.fromarray(alignment))
        reports.append({"image_id": index, "mask_pixels": pixels, "bbox_xywh": bbox,
                        "raycast_vs_render_alpha_iou": iou, "pnp_roundtrip_rotation_error_deg": angle_error,
                        "pnp_roundtrip_translation_error_mm": translation_error,
                        "pnp_roundtrip_max_reprojection_error_px": pnp_pixel_error,
                        "blender_vs_KRT_max_projection_error_px": keypoints["images"][key]["max_projection_error_pixels"],
                        "depth_quantization_max_error_mm": float(np.max(np.abs(encoded[mask].astype(float) * scale - depth_m[mask] * 1000)))})
    save_json(scene / "scene_gt_info.json", info)
    for folder in ("rgb", "depth"):
        assert {path.name for path in (scene / folder).glob("*.png")} == expected_names
    expected_masks = {f"{i:06d}_000000.png" for i in range(20)}
    for folder in ("mask", "mask_visib"):
        assert {path.name for path in (scene / folder).glob("*.png")} == expected_masks
    thumbnail_sheet(annotated_images, titles, scene / "checks/labels_contact_sheet.jpg")
    thumbnail_sheet(rgb_images, titles, scene / "checks/rgb_contact_sheet.jpg")
    thumbnail_sheet(depth_images, titles, scene / "checks/depth_contact_sheet.jpg")
    worst = sorted(range(20), key=lambda i: reports[i]["raycast_vs_render_alpha_iou"])[:4]
    alignment_sheet = Image.new("RGB", (1280, 1024), "#eeeeee")
    for slot, i in enumerate(worst):
        x, y = slot % 2 * 640, slot // 2 * 512
        alignment_sheet.paste(alignment_images[i], (x, y))
        ImageDraw.Draw(alignment_sheet).text((x + 8, y + 486),
            f"{i:06d} / IoU {reports[i]['raycast_vs_render_alpha_iou']:.6f} / green=ray cyan=render", fill="black")
    alignment_sheet.save(scene / "checks/mask_alignment_worst4.jpg", quality=94)
    summary = {"image_count": 20, "resolution": [640, 480], "rotation_translation_and_keypoints": "PASS",
               "png_depth_and_mask": "PASS", "min_ray_render_silhouette_iou": min(r["raycast_vs_render_alpha_iou"] for r in reports),
               "max_KRT_vs_blender_projection_error_px": max(r["blender_vs_KRT_max_projection_error_px"] for r in reports),
               "max_pnp_roundtrip_rotation_error_deg": max(r["pnp_roundtrip_rotation_error_deg"] for r in reports),
               "max_pnp_roundtrip_translation_error_mm": max(r["pnp_roundtrip_translation_error_mm"] for r in reports),
               "max_pnp_roundtrip_reprojection_error_px": max(r["pnp_roundtrip_max_reprojection_error_px"] for r in reports),
               "max_depth_quantization_error_mm": max(r["depth_quantization_max_error_mm"] for r in reports),
               "note": "These are synthetic label consistency tests, not trained-network accuracy measurements.", "frames": reports}
    passed = summary["min_ray_render_silhouette_iou"] >= .99
    summary["status"] = "PASS" if passed else "FAIL_MASK_ALIGNMENT"
    save_json(scene / "checks/validation.json", summary)
    manifest.update(status="verified_smoke_test" if passed else "validation_failed",
                    validated_Asia_Shanghai=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    validation_summary={k: v for k, v in summary.items() if k != "frames"})
    files = [path for folder in ("rgb", "depth", "mask", "mask_visib") for path in sorted((scene / folder).glob("*.png"))]
    files += [scene / name for name in ("scene_camera.json", "scene_gt.json", "scene_gt_info.json", "keypoints_2d.json")]
    manifest["output_sha256"] = {str(path.relative_to(scene)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    save_json(manifest_path, manifest)
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}, indent=2), flush=True)
    if not passed:
        raise SystemExit("Rendered silhouette and BVH mask are not aligned; do not expand this dataset")


if __name__ == "__main__":
    main()
