"""Twenty-view Blender 5.2 BOP smoke-test adapter for the HCCEPose data flow.

This is not the unmodified BlenderProc pipeline. It deliberately uses one
unoccluded object and the already prepared PLY. RGB comes from Cycles; metric
camera-Z depth comes from independent mesh BVH ray casting at pixel centers.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree
import numpy as np


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scene-id", default="000000")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    root = Path(__file__).resolve().parent.parent
    dataset = args.dataset.resolve()
    if not dataset.is_relative_to(root / "datasets") or not (len(args.scene_id) == 6 and args.scene_id.isdigit()):
        raise RuntimeError("Unexpected dataset or scene ID")
    if args.count != 20:
        raise RuntimeError("This smoke test is intentionally limited to 20 images")
    preparation = json.loads((dataset / "metadata/preparation.json").read_text(encoding="utf-8"))
    assert preparation["status"] == "prepared_and_verified"
    for relative in ("models/obj_000001.ply", "models/obj_000001.png", "camera.json", "keypoints_3d.json"):
        digest = hashlib.sha256((dataset / relative).read_bytes()).hexdigest()
        assert digest == preparation["output_hashes"][relative], f"Prepared input changed: {relative}"
    output = dataset / "train_pbr" / args.scene_id
    if output.exists():
        raise RuntimeError("Refusing to overwrite an existing scene")
    for folder in ("rgb", "depth", "mask", "mask_visib", "checks/rgba", "checks/depth_m", "checks/overlay"):
        (output / folder).mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "rendering", "time_Asia_Shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "count": args.count, "seed": args.seed, "obj_id": 1, "scene_id": args.scene_id,
        "backend": "Blender " + bpy.app.version_string + " Cycles CPU + mesh BVH",
        "implementation": "Blender 5.2 smoke-test adapter, not the original BlenderProc script",
        "hccepose_commit": preparation["hccepose_commit"],
        "input_hashes": preparation["output_hashes"], "camera_status": "demo_intrinsics_not_calibrated",
        "pose_convention": "BOP/OpenCV model-to-camera rotation; translation and model coordinates in mm",
        "depth_convention": "optical-axis camera Z, PNG uint16 * depth_scale = mm; 0 means background",
        "keypoint_convention": "all 8 geometric cuboid projections, including hidden corners; not surface-keypoint visibility",
        "limitations": ["one unoccluded instance", "flat randomized backgrounds and area lights; no arm or distractors",
                        "no target-video frames used", "base-color PLY material, not the original full GLB PBR material",
                        "no training or independent real-video accuracy evaluation"], "frames": []}
    manifest_path = output / "run_manifest.json"
    save_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        temporary = output / "checks" / "_temporary_uv_import.ply"
        text = (dataset / "models/obj_000001.ply").read_text(encoding="ascii")
        temporary.write_text(text.replace("property float texture_u", "property float s")
                             .replace("property float texture_v", "property float t"), encoding="ascii")
        del text
        options = {"filepath": str(temporary), "forward_axis": "Y", "up_axis": "Z"}
        if "merge_verts" in bpy.ops.wm.ply_import.get_rna_type().properties:
            options["merge_verts"] = False
        try:
            bpy.ops.wm.ply_import(**options)
        finally:
            temporary.unlink()
        objects = [item for item in bpy.context.scene.objects if item.type == "MESH"]
        assert len(objects) == 1
        obj = objects[0]
        obj.data.transform(Matrix.Scale(.001, 4))
        obj.matrix_world = Matrix.Identity(4)
        assert obj.data.uv_layers.active is not None
        assert len(obj.data.polygons) == preparation["triangle_count"]
        obj.data.polygons.foreach_set("use_smooth", np.ones(len(obj.data.polygons), dtype=bool))
        material = bpy.data.materials.new("BOP base color")
        material.use_nodes = True
        shader = material.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Roughness"].default_value = .7
        shader.inputs["Specular IOR Level"].default_value = .2
        image_node = material.node_tree.nodes.new("ShaderNodeTexImage")
        image_node.image = bpy.data.images.load(str(dataset / "models/obj_000001.png"))
        material.node_tree.links.new(image_node.outputs["Color"], shader.inputs["Base Color"])
        obj.data.materials.clear()
        obj.data.materials.append(material)
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 32
        scene.cycles.use_denoising = False
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 4
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        scene.view_settings.view_transform = "Standard"
        scene.world = bpy.data.worlds.new("Smoke-test environment")
        scene.world.use_nodes = True
        background = scene.world.node_tree.nodes["Background"]
        camera = bpy.data.objects.new("Observing camera", bpy.data.cameras.new("Observing camera"))
        scene.collection.objects.link(camera)
        scene.camera = camera
        intrinsics = json.loads((dataset / "camera.json").read_text())
        width, height = intrinsics["width"], intrinsics["height"]
        fx, fy, cx, cy = [intrinsics[key] for key in ("fx", "fy", "cx", "cy")]
        K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        scene.render.resolution_x, scene.render.resolution_y = width, height
        scene.render.resolution_percentage = 100
        camera.data.sensor_fit, camera.data.sensor_width = "HORIZONTAL", 36
        camera.data.lens = fx * 36 / width
        scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = max(1., fy / fx), max(1., fx / fy)
        aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
        camera.data.shift_x = -(cx - (width - 1) / 2) / width
        camera.data.shift_y = (cy - (height - 1) / 2) / width * aspect
        camera.data.clip_start, camera.data.clip_end = .001, 10
        lights = []
        for i in range(2):
            light = bpy.data.objects.new(f"Area light {i}", bpy.data.lights.new(f"Area light {i}", "AREA"))
            scene.collection.objects.link(light)
            lights.append(light)
        bpy.context.view_layer.update()
        tree = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
        keypoints = json.loads((dataset / "keypoints_3d.json").read_text())
        corners_mm = np.array(keypoints["corners"])
        rng = np.random.default_rng(args.seed)
        cameras, ground_truth, keypoint_labels = {}, {}, {}
        elevations = [-20, 0, 20, 40, 60]
        for index in range(args.count):
            before = time.monotonic()
            azimuth = math.radians(index * 360 / args.count + rng.uniform(-2, 2))
            elevation = math.radians(elevations[(index * 2) % len(elevations)] + rng.uniform(-3, 3))
            radius = float(rng.uniform(.135, .205))
            camera.location = (radius * math.sin(azimuth) * math.cos(elevation),
                               -radius * math.cos(azimuth) * math.cos(elevation), radius * math.sin(elevation))
            target = Vector(rng.uniform(-.004, .004, 3))
            roll = float(rng.uniform(-math.pi / 6, math.pi / 6))
            camera.rotation_mode = "QUATERNION"
            camera.rotation_quaternion = (target - camera.location).to_track_quat("-Z", "Y") @ Quaternion((0, 0, 1), roll)
            bpy.context.view_layer.update()
            world_to_cv = np.diag([1., -1., -1., 1.]) @ np.linalg.inv(np.array(camera.matrix_world))
            R, t_m = world_to_cv[:3, :3], world_to_cv[:3, 3]
            t_mm = t_m * 1000
            camera_points = corners_mm @ R.T + t_mm
            assert np.min(camera_points[:, 2]) > 0
            homogeneous = camera_points @ K.T
            projected = homogeneous[:, :2] / homogeneous[:, 2:]
            assert np.all((projected[:, 0] >= 3) & (projected[:, 0] < width - 3)
                          & (projected[:, 1] >= 3) & (projected[:, 1] < height - 3)), "Cuboid clipped"
            blender_points = []
            for corner in corners_mm:
                point = world_to_camera_view(scene, camera, Vector(corner * .001))
                blender_points.append([point.x * width - .5, (1 - point.y) * height - .5])
            projection_error = float(np.max(np.abs(projected - np.array(blender_points))))
            assert projection_error < .001
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-6) and abs(np.linalg.det(R) - 1) < 1e-6

            x0, y0 = np.floor(projected.min(axis=0) - 2).astype(int)
            x1, y1 = np.ceil(projected.max(axis=0) + 2).astype(int)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1 + 1), min(height, y1 + 1)
            xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            rays_cv = np.column_stack(((xs.ravel() - cx) / fx, (ys.ravel() - cy) / fy, np.ones(xs.size)))
            lengths = np.linalg.norm(rays_cv, axis=1)
            directions = (rays_cv @ R) / lengths[:, None]
            depth = np.zeros((height, width), dtype=np.float32)
            origin = camera.location.copy()
            recovered_surface_error = 0.
            for ray_index, direction in enumerate(directions):
                hit, _, _, distance = tree.ray_cast(origin, Vector(direction), 10.)
                if hit is not None:
                    u, v = int(xs.ravel()[ray_index]), int(ys.ravel()[ray_index])
                    depth[v, u] = distance / lengths[ray_index]
            mask = depth > 0
            assert mask.sum() >= 300
            valid_y, valid_x = np.nonzero(mask)
            sample_ids = np.linspace(0, len(valid_x) - 1, min(64, len(valid_x))).astype(int)
            for sample_id in sample_ids:
                u, v = valid_x[sample_id], valid_y[sample_id]
                # Validate camera-Z depth after the intended 0.1-mm PNG quantization.
                z = round(float(depth[v, u]) * 1000 / intrinsics["depth_scale"]) * intrinsics["depth_scale"] / 1000
                p_cv = np.array([(u - cx) / fx, (v - cy) / fy, 1.]) * z
                p_world = R.T @ (p_cv - t_m)
                _, _, _, surface_distance = tree.find_nearest(Vector(p_world))
                recovered_surface_error = max(recovered_surface_error, surface_distance)
            assert recovered_surface_error < .0001, "Depth back-projection exceeds 0.1 mm from mesh"
            for light_index, light in enumerate(lights):
                light.location = (float(rng.uniform(-.18, .18)), float(rng.uniform(-.18, .18)), float(rng.uniform(.08, .22)))
                light.rotation_euler = (-light.location).to_track_quat("-Z", "Y").to_euler()
                light.data.energy = float(rng.uniform(.3, 1.4))
                light.data.size = float(rng.uniform(.12, .25))
                light.data.color = tuple(float(x) for x in rng.uniform(.8, 1., 3))
            background.inputs["Color"].default_value = tuple(float(x) for x in rng.uniform(.5, .85, 3)) + (1.,)
            background.inputs["Strength"].default_value = float(rng.uniform(.35, .75))
            frame = f"{index:06d}"
            scene.cycles.seed = args.seed + index
            scene.render.filepath = str(output / "checks/rgba" / (frame + ".png"))
            bpy.ops.render.render(write_still=True)
            np.save(output / "checks/depth_m" / (frame + ".npy"), depth)
            key = str(index)
            cameras[key] = {"cam_K": K.reshape(-1).tolist(), "depth_scale": intrinsics["depth_scale"]}
            ground_truth[key] = [{"obj_id": 1, "cam_R_m2c": R.reshape(-1).tolist(), "cam_t_m2c": t_mm.tolist()}]
            keypoint_labels[key] = {"obj_id": 1, "gt_index": 0, "corners_2d": projected.tolist(),
                                    "in_image": [True] * 8, "max_projection_error_pixels": projection_error}
            manifest["frames"].append({"image_id": index, "azimuth_degrees": math.degrees(azimuth),
                "elevation_degrees": math.degrees(elevation), "roll_degrees": math.degrees(roll), "radius_m": radius,
                "target_world_m": list(target), "background_rgb_u8": rng.integers(45, 221, 3).tolist(),
                "raycast_pixels": int(mask.sum()), "depth_backprojection_max_surface_error_mm": recovered_surface_error * 1000,
                "seconds": time.monotonic() - before})
            save_json(output / "scene_camera.json", cameras)
            save_json(output / "scene_gt.json", ground_truth)
            save_json(output / "keypoints_2d.json", {"corner_order_source": "../../keypoints_3d.json",
                      "edges": keypoints["edges"], "images": keypoint_labels})
            save_json(manifest_path, manifest)
            print(f"FRAME {index + 1}/{args.count} rendered; {int(mask.sum())} object pixels; "
                  f"projection error {projection_error:.6f}px; {time.monotonic() - before:.1f}s", flush=True)
        manifest.update(status="rendered_pending_postprocess", render_seconds=time.monotonic() - started)
        save_json(manifest_path, manifest)
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        save_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
