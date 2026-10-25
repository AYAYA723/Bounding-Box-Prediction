"""Render diverse RGB corner data with the prepared HCCEPose camera conventions."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector
import numpy as np


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[2]
    cache = root / ".cache/blender"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["OPTIX_CACHE_PATH"] = str(cache)
    os.environ["CUDA_CACHE_PATH"] = str(cache)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--device", choices=["CPU", "OPTIX"], default="CPU")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    dataset = root / "datasets/action4"
    preparation = json.loads((dataset / "metadata/preparation.json").read_text())
    source = Path(preparation["source"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == preparation["source_sha256"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "rgba").mkdir()
    keypoints = json.loads((dataset / "keypoints_3d.json").read_text())
    intrinsics = json.loads((dataset / "camera.json").read_text())
    save_json(args.output / "keypoints_3d.json", keypoints)
    corners_mm = np.array(keypoints["corners"])
    width, height = intrinsics["width"], intrinsics["height"]
    fx, fy, cx, cy = [intrinsics[key] for key in ("fx", "fy", "cx", "cy")]
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    rng = np.random.default_rng(args.seed)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source), merge_vertices=False)
    transform = Matrix(preparation["source_blender_world_to_bop_mm"])
    transform = Matrix.Scale(.001, 4) @ transform
    for obj in list(bpy.context.scene.objects):
        if obj.type == "MESH":
            world = obj.matrix_world.copy()
            obj.parent = None
            obj.data.transform(transform @ world)
            obj.matrix_world = Matrix.Identity(4)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU" if args.device == "CPU" else "GPU"
    if args.device == "OPTIX":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "OPTIX"
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type == "OPTIX"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = False
    scene.render.use_persistent_data = True
    scene.cycles.max_bounces = 3
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.world = bpy.data.worlds.new("Varied environment")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    camera = bpy.data.objects.new("Observing camera", bpy.data.cameras.new("Observing camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.sensor_fit, camera.data.sensor_width = "HORIZONTAL", 36
    camera.data.lens = fx * 36 / width
    scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = max(1., fy / fx), max(1., fx / fy)
    aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    camera.data.shift_x = -(cx - (width - 1) / 2) / width
    camera.data.shift_y = (cy - (height - 1) / 2) / width * aspect
    camera.data.clip_start, camera.data.clip_end = .001, 10
    lights = []
    for index in range(2):
        light = bpy.data.objects.new(f"Area light {index}", bpy.data.lights.new(f"Area light {index}", "AREA"))
        scene.collection.objects.link(light)
        lights.append(light)

    cameras, poses, labels, records = {}, {}, {}, []
    started = time.monotonic()
    manifest = {"status": "rendering", "seed": args.seed, "count": args.count,
                "asset_id": source.stem, "source_sha256": preparation["source_sha256"],
                "renderer": "Blender " + bpy.app.version_string + f" Cycles {args.device}, 32 samples, original GLB PBR materials",
                "hccepose_commit": preparation["hccepose_commit"],
                "implementation": "Blender adapter following the existing HCCEPose/BOP coordinate flow; RGB-only corner dataset",
                "pose_convention": "OpenCV model-to-camera; translation and model coordinates in mm",
                "camera_intrinsics": "existing HCCEPose demo intrinsics, not calibrated to the test video",
                "source_video_used_for_training": False, "frames": records}
    for index in range(args.count):
        before = time.monotonic()
        for attempt in range(1000):
            azimuth = 2 * math.pi * (index + rng.random()) / args.count
            elevation = math.radians(float(rng.uniform(-75, 75)))
            radius = float(np.exp(rng.uniform(np.log(.10), np.log(.32))))
            roll = float(rng.uniform(-math.pi, math.pi))
            camera.location = (radius * math.sin(azimuth) * math.cos(elevation),
                               -radius * math.cos(azimuth) * math.cos(elevation), radius * math.sin(elevation))
            camera.rotation_mode = "QUATERNION"
            camera.rotation_quaternion = (-camera.location).to_track_quat("-Z", "Y") @ Quaternion((0, 0, 1), roll)
            camera.rotation_quaternion.normalize()
            desired_x, desired_y = rng.uniform(.18 * width, .82 * width), rng.uniform(.18 * height, .82 * height)
            shift = Vector(((cx - desired_x) * radius / fx, (desired_y - cy) * radius / fy, 0))
            camera.location += camera.rotation_quaternion @ shift
            bpy.context.view_layer.update()
            world_to_cv = np.diag([1., -1., -1., 1.]) @ np.linalg.inv(np.array(camera.matrix_world))
            u, _, vt = np.linalg.svd(world_to_cv[:3, :3])
            R = u @ vt
            t_mm = -R @ np.array(camera.location, dtype=np.float64) * 1000
            camera_points = corners_mm @ R.T + t_mm
            homogeneous = camera_points @ K.T
            projected = homogeneous[:, :2] / homogeneous[:, 2:]
            size = np.ptp(projected, axis=0)
            inside = np.all(projected >= [24, 24]) and np.all(projected <= [width - 25, height - 25])
            corner_radius = np.linalg.norm(projected - projected.mean(axis=0), axis=1).min()
            if camera_points[:, 2].min() > 0 and inside and 80 <= size.max() <= 380 and corner_radius >= 5:
                break
        else:
            raise RuntimeError("Could not sample an in-frame camera pose")
        independent = []
        for corner in corners_mm:
            point = world_to_camera_view(scene, camera, Vector(corner * .001))
            independent.append([point.x * width - .5, (1 - point.y) * height - .5])
        error = float(np.max(np.abs(projected - independent)))
        assert error < .001
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6) and abs(np.linalg.det(R) - 1) < 1e-6
        for light in lights:
            light.location = tuple(float(v) for v in rng.uniform(-.2, .2, 3))
            light.rotation_euler = (-light.location).to_track_quat("-Z", "Y").to_euler()
            light.data.energy = float(rng.uniform(.25, 1.6))
            light.data.size = float(rng.uniform(.1, .25))
            light.data.color = tuple(float(v) for v in rng.uniform(.75, 1., 3))
        background.inputs["Color"].default_value = tuple(float(v) for v in rng.uniform(.5, .9, 3)) + (1.,)
        background.inputs["Strength"].default_value = float(rng.uniform(.25, .85))
        scene.cycles.seed = args.seed + index
        scene.render.filepath = str(args.output / "rgba" / f"{index:06d}.png")
        bpy.ops.render.render(write_still=True)
        key = str(index)
        cameras[key] = {"cam_K": K.reshape(-1).tolist()}
        poses[key] = [{"obj_id": 1, "cam_R_m2c": R.reshape(-1).tolist(), "cam_t_m2c": t_mm.tolist()}]
        labels[key] = {"obj_id": 1, "gt_index": 0, "corners_2d": projected.tolist(), "in_image": [True] * 8,
                       "max_projection_error_pixels": error}
        records.append({"image_id": index, "asset_id": source.stem, "azimuth_degrees": math.degrees(azimuth),
                        "elevation_degrees": math.degrees(elevation), "roll_degrees": math.degrees(roll),
                        "radius_m": radius, "projected_size_wh": size.tolist(),
                        "seconds": time.monotonic() - before})
        if (index + 1) % 25 == 0 or index + 1 == args.count:
            save_json(args.output / "scene_camera.json", cameras)
            save_json(args.output / "scene_gt.json", poses)
            save_json(args.output / "keypoints_2d.json", {"corner_order_source": "keypoints_3d.json",
                      "edges": keypoints["edges"], "images": labels})
            manifest["render_seconds"] = time.monotonic() - started
            save_json(args.output / "render_manifest.json", manifest)
            print(f"PROGRESS {index + 1}/{args.count}, elapsed {time.monotonic() - started:.1f}s", flush=True)
    manifest["status"] = "rendered"
    save_json(args.output / "render_manifest.json", manifest)


if __name__ == "__main__":
    main()
