"""Build eight-channel training targets using BoxDreamer section 3.1.

The paper's kernel is exp(-distance / (corner_to_center_distance / 10)**2),
NOT a Gaussian with squared distance in the numerator. This implementation
keeps its [0, 1] values, matching the paper's Sigmoid output. Upstream's
additional peak normalization and [-1, 1] remapping are deliberately absent.
See heatmaps_README.md for references, coordinate conventions and usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = PROJECT / "datasets/action4/train_pbr/000000"
PAPER_URL = "https://arxiv.org/html/2504.07955v2#S3.SS1"
CODE_URL = (
    "https://github.com/zju3dv/BoxDreamer/blob/main/"
    "src/datasets/utils/base/bbox_utils.py"
)
COLORS = np.array(
    [[255, 90, 90], [255, 170, 40], [240, 220, 60], [80, 220, 120],
     [30, 210, 235], [80, 140, 255], [185, 110, 255], [255, 110, 205]],
    dtype=np.uint8,
)


def make_corner_heatmaps(
    corners_xy: np.ndarray, image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (heatmaps[8,H,W], center_xy[2], denominators[8]).

    image_size is (width, height), following PIL. Coordinates are (x, y),
    with integer pixel centers x=0..W-1, y=0..H-1. Subpixel coordinates are
    preserved. Occluded and off-image corners are not clipped or discarded.
    The center is the mean of the eight projected corners, as in upstream
    make_bbox_features; it is not the projection of the 3D center.
    """
    corners = np.asarray(corners_xy, dtype=np.float64)
    if corners.shape != (8, 2) or not np.isfinite(corners).all():
        raise ValueError("corners_xy must be a finite (8, 2) array in (x, y) order")
    if len(image_size) != 2 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0
        for v in image_size
    ):
        raise ValueError("image_size must contain positive integer (width, height)")
    width, height = image_size
    center = corners.mean(axis=0)
    # Paper: 2*sigma_i**2 = (||corner_i - center||_2 / 10)**2.
    with np.errstate(over="ignore", invalid="ignore"):
        denominators = np.sum((corners - center) ** 2, axis=1) / 100.0
    if not np.isfinite(denominators).all() or np.any(denominators <= 0):
        raise ValueError("Each corner must have a finite, nonzero distance to the 2D center")
    xs = np.arange(width, dtype=np.float64)[None, :]
    ys = np.arange(height, dtype=np.float64)[:, None]
    heatmaps = np.empty((8, height, width), dtype=np.float32)
    for channel, ((x, y), denominator) in enumerate(zip(corners, denominators)):
        distances = np.hypot(xs - x, ys - y)
        heatmaps[channel] = np.exp(-distances / denominator)
    return heatmaps, center, denominators


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_heatmaps(
    heatmaps: np.ndarray, corners_xy: np.ndarray, center_xy: np.ndarray,
    denominators: np.ndarray,
) -> dict:
    """Validate all peaks and independently evaluate scalar kernel samples."""
    if heatmaps.ndim != 3 or heatmaps.shape[0] != 8 or heatmaps.dtype != np.float32:
        raise ValueError("Expected float32 [8, H, W] heatmaps")
    if not np.isfinite(heatmaps).all() or heatmaps.min() < 0 or heatmaps.max() > 1:
        raise ValueError("Heatmaps must contain finite values in [0, 1]")
    _, height, width = heatmaps.shape
    nearest = np.floor(corners_xy + 0.5).astype(np.int64)
    nearest[:, 0] = np.clip(nearest[:, 0], 0, width - 1)
    nearest[:, 1] = np.clip(nearest[:, 1], 0, height - 1)
    flat_indices = heatmaps.reshape(8, -1).argmax(axis=1)
    peaks_xy = np.stack((flat_indices % width, flat_indices // width), axis=1)
    errors = np.linalg.norm(peaks_xy - corners_xy, axis=1)
    sample_error = 0.0
    for channel, ((x, y), (px, py)) in enumerate(zip(corners_xy, peaks_xy)):
        # Ties are possible at half pixels: compare the achieved distance.
        nearest_distance = float(np.linalg.norm(nearest[channel] - corners_xy[channel]))
        if abs(float(errors[channel]) - nearest_distance) > 1e-6:
            raise ValueError(f"Channel {channel}: heatmap maximum is not nearest to its corner")
        # Scalar math is separate from the vectorized generation path.
        import math
        denominator = (math.hypot(x - center_xy[0], y - center_xy[1]) / 10) ** 2
        if not math.isclose(float(denominators[channel]), denominator, rel_tol=1e-12):
            raise ValueError("Incorrect adaptive kernel denominator")
        positions = [(0, 0), (width - 1, height - 1), (width // 2, height // 2), (int(px), int(py))]
        for sx, sy in positions:
            expected = math.exp(-math.hypot(sx - x, sy - y) / denominator)
            error = abs(float(heatmaps[channel, sy, sx]) - expected)
            sample_error = max(sample_error, error)
            if error > 6e-8:
                raise ValueError(f"Channel {channel}: value differs from the paper formula")
    inside = (
        (corners_xy[:, 0] >= 0) & (corners_xy[:, 0] < width)
        & (corners_xy[:, 1] >= 0) & (corners_xy[:, 1] < height)
    )
    return {
        "shape": list(heatmaps.shape), "dtype": str(heatmaps.dtype),
        "range": [float(heatmaps.min()), float(heatmaps.max())],
        "peak_xy": peaks_xy.tolist(), "peak_errors_pixels": errors.tolist(),
        "in_image": inside.tolist(),
        "max_scalar_formula_error": sample_error,
    }


def draw_alignment(rgb: Image.Image, corners: np.ndarray, peaks: np.ndarray, edges: list) -> Image.Image:
    canvas = rgb.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for a, b in edges:
        draw.line((tuple(corners[a]), tuple(corners[b])), fill="white", width=1)
    for channel, ((x, y), (px, py)) in enumerate(zip(corners, peaks)):
        color = tuple(int(c) for c in COLORS[channel])
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
        draw.line((int(px) - 3, int(py), int(px) + 3, int(py)), fill="white", width=1)
        draw.line((int(px), int(py) - 3, int(px), int(py) + 3), fill="white", width=1)
        draw.text((x + 6, y + 4), str(channel), fill=color, stroke_width=1, stroke_fill="black")
    return canvas


def draw_channels(rgb: Image.Image, heatmaps: np.ndarray, corners: np.ndarray, peaks: np.ndarray) -> Image.Image:
    """Eight channels over RGB; color strength is the unnormalized label value."""
    width, height = rgb.size
    tile_w, tile_h = min(width, 480), round(height * min(width, 480) / width)
    sheet = Image.new("RGB", (4 * tile_w, 2 * (tile_h + 34)), "#eeeeee")
    source = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    for channel in range(8):
        weights = heatmaps[channel, :, :, None] * 0.78
        tinted = np.rint(source * (1 - weights) + COLORS[channel] * weights).clip(0, 255).astype(np.uint8)
        tile = Image.fromarray(tinted)
        draw = ImageDraw.Draw(tile)
        x, y = corners[channel]
        px, py = peaks[channel]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="white", width=2)
        draw.line((int(px) - 3, int(py), int(px) + 3, int(py)), fill="black", width=2)
        draw.line((int(px), int(py) - 3, int(px), int(py) + 3), fill="black", width=2)
        left, top = channel % 4 * tile_w, channel // 4 * (tile_h + 34)
        sheet.paste(tile.resize((tile_w, tile_h), Image.Resampling.LANCZOS), (left, top))
        ImageDraw.Draw(sheet).text(
            (left + 6, top + tile_h + 5),
            f"corner {channel} / peak {heatmaps[channel].max():.6f}", fill="black",
        )
    return sheet


def generate_scene(scene: Path, output: Path) -> dict:
    scene, output = scene.resolve(), output.resolve()
    labels_path = scene / "keypoints_2d.json"
    rgb_dir = scene / "rgb"
    source = json.loads(labels_path.read_text(encoding="utf-8"))
    images = source.get("images", {})
    if not images:
        raise ValueError("No image annotations in keypoints_2d.json")
    # Complete input preflight before creating the output directory.
    inputs = []
    for key in sorted(images, key=int):
        image_id = int(key)
        if image_id < 0 or str(image_id) != key:
            raise ValueError(f"Invalid image ID: {key}")
        entry = images[key]
        corners = np.asarray(entry["corners_2d"], dtype=np.float64)
        if corners.shape != (8, 2) or not np.isfinite(corners).all():
            raise ValueError(f"Invalid corners for frame {key}")
        rgb_path = rgb_dir / f"{image_id:06d}.png"
        with Image.open(rgb_path) as image:
            image_size = image.size
        inputs.append((image_id, entry, corners, rgb_path, image_size))
    output.mkdir(parents=True, exist_ok=False)
    checks = output / "checks"
    checks.mkdir()
    tiles, frames = [], []
    source_hashes = {"keypoints_2d.json": sha256(labels_path)}
    for image_id, entry, corners, rgb_path, image_size in inputs:
        stem = f"{image_id:06d}"
        heatmaps, center, denominators = make_corner_heatmaps(corners, image_size)
        report = check_heatmaps(heatmaps, corners, center, denominators)
        inside = np.asarray(report["in_image"], dtype=bool)
        if "in_image" in entry and not np.array_equal(inside, entry["in_image"]):
            raise ValueError(f"in_image flags disagree for frame {image_id}")
        target_path = output / f"{stem}.npz"
        np.savez_compressed(
            target_path, heatmaps=heatmaps, corners_xy=corners, center_xy=center,
            denominators=denominators, in_image=inside,
            image_size_wh=np.asarray(image_size, dtype=np.int32),
            image_id=np.int64(image_id), obj_id=np.int64(entry["obj_id"]),
            gt_index=np.int64(entry["gt_index"]),
        )
        # Read the actual serialized targets, rather than validating memory only.
        with np.load(target_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["heatmaps"], heatmaps):
                raise ValueError(f"Serialization changed heatmaps: {target_path}")
            if not np.array_equal(saved["corners_xy"], corners):
                raise ValueError("Serialization changed corner order or coordinates")
        rgb_hash = sha256(rgb_path)
        source_hashes[f"rgb/{stem}.png"] = rgb_hash
        with Image.open(rgb_path) as image:
            rgb = image.convert("RGB")
        peaks = np.asarray(report["peak_xy"], dtype=np.int64)
        if not frames:
            draw_channels(rgb, heatmaps, corners, peaks).save(checks / f"{stem}_eight_channels.jpg", quality=94)
        alignment = draw_alignment(rgb, corners, peaks, source.get("edges", []))
        tile_w = 320
        tile_h = round(rgb.height * tile_w / rgb.width)
        tile = Image.new("RGB", (tile_w, tile_h + 30), "#eeeeee")
        tile.paste(alignment.resize((tile_w, tile_h), Image.Resampling.LANCZOS))
        ImageDraw.Draw(tile).text(
            (6, tile_h + 6), f"{stem} / max peak error {max(report['peak_errors_pixels']):.3f} px", fill="black",
        )
        tiles.append(tile)
        frames.append({
            "image_id": image_id, "obj_id": entry["obj_id"], "gt_index": entry["gt_index"],
            "rgb": str(rgb_path.relative_to(scene).as_posix()), "heatmap": target_path.name,
            "heatmap_sha256": sha256(target_path), "size_bytes": target_path.stat().st_size,
            "center_xy": center.tolist(), "denominators": denominators.tolist(), **report,
        })
    cols = min(5, len(tiles))
    row_height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (cols * 320, ((len(tiles) + cols - 1) // cols) * row_height), "#eeeeee")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index % cols * 320, index // cols * row_height))
    sheet.save(checks / "peak_alignment_contact_sheet.jpg", quality=94)
    for relative, digest in source_hashes.items():
        if sha256(scene / relative) != digest:
            raise ValueError(f"Source data changed during generation: {relative}")
    validation = {
        "status": "PASS", "image_count": len(frames), "channel_count": 8,
        "max_peak_error_pixels": max(max(frame["peak_errors_pixels"]) for frame in frames),
        "max_scalar_formula_error": max(frame["max_scalar_formula_error"] for frame in frames),
        "out_of_image_corner_count": sum(not flag for frame in frames for flag in frame["in_image"]),
        "all_peak_locations_are_nearest_grid_points": True,
        "serialized_arrays_checked": True, "input_files_unchanged": True,
        "total_label_bytes": sum(frame["size_bytes"] for frame in frames),
        "note": "This verifies target generation, not trained-network prediction accuracy.",
    }
    write_json(checks / "validation.json", validation)
    manifest = {
        "schema_version": 1, "paper": PAPER_URL, "upstream_center_reference": CODE_URL,
        "mode": "paper_formula_0_to_1", "layout": "CHW", "dtype": "float32",
        "formula": "H[i,y,x] = exp(-hypot(x-x_i,y-y_i) / (hypot(x_i-c_x,y_i-c_y)/10)^2)",
        "center_definition": "arithmetic mean of all eight projected (x,y) corners",
        "pixel_coordinates": "origin at top-left pixel center; x=column, y=row; no resize or crop",
        "channel_order": "unchanged indices 0..7 from keypoints_2d.json and keypoints_3d.json",
        "corner_order_source": source.get("corner_order_source"),
        "visibility_policy": "all eight virtual box corners, including occluded ones",
        "off_image_policy": "evaluate kernel at original coordinate; no clamping; inspect in_image when using coordinate loss",
        "normalization": "none; not sum-normalized, not max-normalized; sampled subpixel peak can be below 1",
        "upstream_difference": "official make_bbox_features additionally divides by maximum and maps to [-1,1]; omitted to follow paper/Sigmoid description",
        "script_sha256": sha256(Path(__file__)),
        "input_scene": str(scene), "input_sha256": source_hashes,
        "validation": validation, "frames": frames,
    }
    write_json(output / "manifest.json", manifest)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE,
                        help="BOP scene with rgb/ and keypoints_2d.json")
    parser.add_argument("--output", type=Path,
                        help="new output directory; default: SCENE/heatmaps_boxdreamer")
    args = parser.parse_args()
    output = args.output if args.output is not None else args.scene / "heatmaps_boxdreamer"
    report = generate_scene(args.scene, output)
    print(json.dumps({"output": str(output.resolve()), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
