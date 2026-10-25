"""Round-trip the BOP PLY through Blender and verify perspective projections."""
import argparse
import json
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
dataset = args.dataset.resolve()
checks = dataset / "checks"
manifest_path = dataset / "metadata/preparation.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest["status"] == "geometry_metadata_validated_pending_preview"
bpy.ops.wm.read_factory_settings(use_empty=True)
# Same UV-property aliasing as HCCEPose's bundled ObjectLoader; Blender 5.2
# uses wm.ply_import instead of Blender 3.6's import_mesh.ply.
temporary = checks / "_uv_import_temporary.ply"
content = (dataset / "models/obj_000001.ply").read_text(encoding="ascii")
temporary.write_text(content.replace("property float texture_u", "property float s")
                    .replace("property float texture_v", "property float t"), encoding="ascii")
del content
options = {"filepath": str(temporary), "forward_axis": "Y", "up_axis": "Z"}
if "merge_verts" in bpy.ops.wm.ply_import.get_rna_type().properties:
    options["merge_verts"] = False
try:
    bpy.ops.wm.ply_import(**options)
finally:
    temporary.unlink()
objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
assert len(objects) == 1
obj = objects[0]
mesh = obj.data
assert mesh.uv_layers.active is not None
assert len(mesh.polygons) == manifest["triangle_count"]
position = np.empty((len(mesh.vertices), 3), dtype=np.float64)
mesh.vertices.foreach_get("co", position.ravel())
info = json.loads((dataset / "models/models_info.json").read_text())["1"]
assert np.allclose(position.min(axis=0), [info["min_" + a] for a in "xyz"], atol=1e-5)
assert np.allclose(position.max(axis=0), [info["max_" + a] for a in "xyz"], atol=1e-5)
obj.scale = (.001, .001, .001)
mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
material = bpy.data.materials.new("BOP base-color texture")
material.use_nodes = True
nodes = material.node_tree.nodes
shader = nodes.get("Principled BSDF")
shader.inputs["Roughness"].default_value = .7
texture = nodes.new("ShaderNodeTexImage")
texture.image = bpy.data.images.load(str(dataset / "models/obj_000001.png"))
material.node_tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
mesh.materials.clear()
mesh.materials.append(material)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 16
scene.render.threads_mode = "FIXED"
scene.render.threads = 4
scene.world = bpy.data.worlds.new("Check world")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (.65, .65, .65, 1)
scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = .8
scene.view_settings.view_transform = "Standard"
camera = bpy.data.objects.new("Observing camera", bpy.data.cameras.new("Observing camera"))
scene.collection.objects.link(camera)
scene.camera = camera
intrinsics = json.loads((dataset / "camera.json").read_text())
width, height = intrinsics["width"], intrinsics["height"]
fx, fy, cx, cy = [intrinsics[name] for name in ("fx", "fy", "cx", "cy")]
scene.render.resolution_x, scene.render.resolution_y = width, height
scene.render.resolution_percentage = 100
camera.data.sensor_fit = "HORIZONTAL"
camera.data.sensor_width = 36
camera.data.lens = fx * 36 / width
scene.render.pixel_aspect_x = max(1, fy / fx)
scene.render.pixel_aspect_y = max(1, fx / fy)
aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
camera.data.shift_x = -(cx - (width - 1) / 2) / width
camera.data.shift_y = (cy - (height - 1) / 2) / width * aspect
camera.data.clip_start, camera.data.clip_end = .001, 10
for i, location in enumerate(((.10, -.15, .20), (-.10, .15, .15))):
    light = bpy.data.objects.new("Check light " + str(i), bpy.data.lights.new("Check light", "AREA"))
    scene.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (-light.location).to_track_quat("-Z", "Y").to_euler()
    light.data.energy = 3
    light.data.size = .2
keypoints = json.loads((dataset / "keypoints_3d.json").read_text())
corners = np.array(keypoints["corners"], dtype=np.float64)
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
views = [("front", (.025, -.15, .035)), ("back", (-.025, .15, .035)),
         ("side", (.15, -.015, .025)), ("top_oblique", (.08, -.12, .10))]
results = []
for index, (name, location) in enumerate(views):
    camera.location = location
    camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()
    world_to_cv = np.diag([1., -1., -1., 1.]) @ np.linalg.inv(np.array(camera.matrix_world))
    R, t_mm = world_to_cv[:3, :3], world_to_cv[:3, 3] * 1000
    camera_points = corners @ R.T + t_mm
    assert np.all(camera_points[:, 2] > 0)
    homogeneous = camera_points @ K.T
    projected = homogeneous[:, :2] / homogeneous[:, 2:]
    blender_projected = []
    for corner in corners:
        point = world_to_camera_view(scene, camera, Vector(corner * .001))
        blender_projected.append([point.x * width - .5, (1 - point.y) * height - .5])
    error = float(np.max(np.abs(projected - np.array(blender_projected))))
    assert error < .001, f"Projection mismatch: {error} pixels"
    image_name = f"{index:02d}_{name}.png"
    scene.render.filepath = str(checks / image_name)
    bpy.ops.render.render(write_still=True)
    results.append({"name": name, "image": image_name, "K": K.tolist(), "cam_R_m2c": R.reshape(-1).tolist(),
                    "cam_t_m2c": t_mm.tolist(), "corners_2d": projected.tolist(),
                    "max_projection_error_pixels": error, "pixel_convention": "OpenCV pixel centers"})
(checks / "projection_check.json").write_text(json.dumps({"views": results, "edges": keypoints["edges"],
    "purpose": "Model preparation and coordinate check; not a completed training dataset"}, indent=2), encoding="utf-8")
manifest["status"] = "prepared_and_verified"
manifest["validation"].update({"blender_ply_roundtrip": "PASS", "uv_layer_imported": True,
    "perspective_projection": "PASS", "max_projection_error_pixels": max(r["max_projection_error_pixels"] for r in results),
    "preview_note": "Blender 5.2 PLY import with HCCEPose UV aliases; full BlenderProc training-data generation not run"})
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("PLY roundtrip and four perspective projection checks: PASS", flush=True)
