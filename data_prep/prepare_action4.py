"""Prepare one textured BOP object in a separate, background Blender process."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import struct
import sys

import bpy
import numpy as np
from mathutils import Vector


def write_json(path, value):
    serialized = json.dumps(value, ensure_ascii=False, indent=2)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(serialized):
            raise RuntimeError(f"Refusing to overwrite different metadata: {path}")
        return
    path.write_text(serialized, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--resume-own-export", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    project = Path(__file__).resolve().parent.parent
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_relative_to(project) or not output.is_relative_to(project / "datasets"):
        raise RuntimeError("Inputs and outputs must stay inside this S-drive project")
    if output.exists() and not args.resume_own_export:
        raise RuntimeError("Refusing to overwrite an existing dataset")
    source_bytes = source.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    json_size, _ = struct.unpack_from("<II", source_bytes, 12)
    doc = json.loads(source_bytes[20:20 + json_size])
    binary = memoryview(source_bytes)[28 + json_size:]
    assert len(doc["meshes"]) == len(doc["materials"]) == 1
    texture_index = doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    image_index = doc["textures"][texture_index]["source"]
    image = doc["images"][image_index]
    assert image["mimeType"] == "image/png"
    view = doc["bufferViews"][image["bufferView"]]
    offset = view.get("byteOffset", 0)
    texture_bytes = bytes(binary[offset:offset + view["byteLength"]])
    assert texture_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source), merge_vertices=False)
    objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    assert len(objects) == 1
    obj = objects[0]
    mesh = obj.data
    assert all(len(poly.vertices) == 3 for poly in mesh.polygons)
    vertices = np.empty((len(mesh.vertices), 3), dtype=np.float64)
    mesh.vertices.foreach_get("co", vertices.ravel())
    normals = np.empty_like(vertices)
    mesh.vertices.foreach_get("normal", normals.ravel())
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    world_vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    world_normals = normals @ np.linalg.inv(matrix[:3, :3])
    loop_vertices = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertices)
    loop_uv = np.empty((len(mesh.loops), 2), dtype=np.float64)
    mesh.uv_layers.active.data.foreach_get("uv", loop_uv.ravel())
    vertex_uv = np.zeros((len(vertices), 2), dtype=np.float64)
    vertex_uv[loop_vertices] = loop_uv
    uv_error = float(np.max(np.abs(vertex_uv[loop_vertices] - loop_uv)))
    if uv_error > 1e-6:
        raise RuntimeError("UV seams need explicit vertex splitting; refusing a lossy export")
    faces = loop_vertices.reshape(-1, 3)
    original_vertex_count, original_face_count = len(vertices), len(faces)
    edge1 = world_vertices[faces[:, 1]] - world_vertices[faces[:, 0]]
    edge2 = world_vertices[faces[:, 2]] - world_vertices[faces[:, 0]]
    face_cross = np.cross(edge1, edge2)
    valid_faces = np.einsum("ij,ij->i", face_cross, face_cross) > 1e-28
    faces = faces[valid_faces]
    used = np.unique(faces)
    remap = np.full(original_vertex_count, -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    faces = remap[faces]
    world_vertices = world_vertices[used]
    world_normals = world_normals[used]
    vertex_uv = vertex_uv[used]
    normal_lengths = np.linalg.norm(world_normals, axis=1)
    zero_normals = normal_lengths < 1e-12
    if zero_normals.any():
        geometric_normals = np.zeros_like(world_normals)
        for corner in range(3):
            np.add.at(geometric_normals, faces[:, corner], face_cross[valid_faces])
        world_normals[zero_normals] = geometric_normals[zero_normals]
        normal_lengths = np.linalg.norm(world_normals, axis=1)
    if np.any(normal_lengths < 1e-12):
        raise RuntimeError("Non-surface or contradictory normals remain after cleanup")
    world_normals /= normal_lengths[:, None]
    lower, upper = world_vertices.min(axis=0), world_vertices.max(axis=0)
    center = (lower + upper) / 2
    factor = 70.5 / (upper[0] - lower[0])
    vertices_mm = (world_vertices - center) * factor
    assert np.isfinite(vertices_mm).all() and np.isfinite(world_normals).all() and np.isfinite(vertex_uv).all()
    assert faces.min() >= 0 and faces.max() < len(vertices_mm)

    output.mkdir(parents=True, exist_ok=args.resume_own_export)
    models = output / "models"
    models.mkdir(exist_ok=args.resume_own_export)
    metadata = output / "metadata"
    metadata.mkdir(exist_ok=args.resume_own_export)
    (output / "checks").mkdir(exist_ok=args.resume_own_export)
    texture_path = models / "obj_000001.png"
    if texture_path.exists():
        assert texture_path.read_bytes() == texture_bytes
    else:
        texture_path.write_bytes(texture_bytes)
    ply_path = models / "obj_000001.ply"
    if not ply_path.exists():
        with ply_path.open("w", encoding="ascii", newline="\n") as target:
            target.write("ply\nformat ascii 1.0\ncomment TextureFile obj_000001.png\n"
                         "comment Coordinates are millimeters; origin is the full model AABB center.\n")
            target.write(f"element vertex {len(vertices_mm)}\n")
            for prop in ("x", "y", "z", "nx", "ny", "nz", "texture_u", "texture_v"):
                target.write(f"property float {prop}\n")
            target.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
            np.savetxt(target, np.column_stack((vertices_mm, world_normals, vertex_uv)), fmt="%.9g")
            np.savetxt(target, np.column_stack((np.full(len(faces), 3), faces)), fmt="%d")

    minimum, maximum = vertices_mm.min(axis=0), vertices_mm.max(axis=0)
    corner_signs = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    half = (maximum - minimum) / 2
    write_json(output / "keypoints_3d.json", {
        "obj_id": 1, "units": "mm", "type": "full_object_axis_aligned_bounding_box_including_lens",
        "coordinate_frame": {"x": "right when looking at the front", "y": "toward the back", "z": "up",
                             "front_face_normal": [0, -1, 0], "origin": "full model AABB center"},
        "corner_order_signs_xyz": corner_signs,
        "corners": (np.array(corner_signs) * half).tolist(),
        "edges": [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                  [0, 4], [1, 5], [2, 6], [3, 7]], "front_face_indices": [0, 1, 5, 4]})
    camera = {"cx": 325.2611083984375, "cy": 242.04899588216654, "depth_scale": 0.1,
              "fx": 572.411363389757, "fy": 573.5704328585578, "height": 480, "width": 640}
    write_json(output / "camera.json", camera)
    write_json(metadata / "camera_provenance.json", {
        "status": "demo_intrinsics_not_calibrated", "source": "HCCEPose s2_p1_gen_pbr_data.py defaults",
        "purpose": "Rendering pipeline smoke test only", "measured_from_target_video": False,
        "note": "These are the observing/rendering camera intrinsics, not the Action 4 target object's lens specs."})
    transform = np.eye(4)
    transform[:3, :3] *= factor
    transform[:3, 3] = -center * factor
    # Read the freshly cloned checkout's revision without changing Git trust settings.
    git_head = (args.repo / ".git/HEAD").read_text(encoding="ascii").strip()
    if git_head.startswith("ref: "):
        ref = git_head[5:]
        if not ref.startswith("refs/heads/") or ".." in ref:
            raise RuntimeError("Unexpected Git HEAD reference")
        repo_commit = (args.repo / ".git" / ref).read_text(encoding="ascii").strip()
    else:
        repo_commit = git_head
    assert len(repo_commit) == 40 and all(c in "0123456789abcdef" for c in repo_commit)
    manifest = {
        "status": "exported_pending_validation", "prepared_Asia_Shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "source": str(source), "source_sha256": source_sha,
        "generation_service": "Tencent Hunyuan3D, per user; exact generation settings and input views not recorded",
        "hccepose_commit": repo_commit, "blender": bpy.app.version_string,
        "scale_policy": "uniform; anchor generated model width to 70.5 mm; no geometric stretching or decimation",
        "official_reference_dimensions_width_height_depth_mm": [70.5, 44.2, 32.8],
        "dimension_source": "https://www.dji.com/osmo-action-4/specs",
        "measured_size_xyz_mm": (maximum - minimum).tolist(),
        "measured_width_height_depth_mm": [float(maximum[i] - minimum[i]) for i in (0, 2, 1)],
        "source_blender_world_bounds": [lower.tolist(), upper.tolist()],
        "source_blender_world_to_bop_mm": transform.tolist(),
        "vertex_count": len(vertices_mm), "triangle_count": len(faces),
        "cleanup": {"unreferenced_vertices_removed": original_vertex_count - len(used),
                    "degenerate_faces_removed": original_face_count - len(faces),
                    "zero_normals_repaired_on_used_vertices": int(zero_normals.sum())},
        "max_per_vertex_uv_conflict": uv_error,
        "texture": {"file": "models/obj_000001.png", "dimensions": list(struct.unpack_from(">II", texture_bytes, 16)),
                    "sha256": hashlib.sha256(texture_bytes).hexdigest(), "source": "embedded GLB base-color PNG, byte-for-byte extraction"},
        "material_note": "PLY retains UV/base color and vertex normals. Original normal/metallic/roughness maps remain in the unchanged source GLB; PLY is not a full PBR container.",
        "symmetry": "Whole camera treated as asymmetric; circular lens is not a symmetry of the complete object.",
        "validation": {"source_unchanged": hashlib.sha256(source.read_bytes()).hexdigest() == source_sha,
                       "finite_geometry": True, "valid_triangle_indices": True, "uv_preserved": True}}
    assert manifest["validation"]["source_unchanged"]
    write_json(metadata / "preparation.json", manifest)
    print(json.dumps({"output": str(output), "vertices": len(vertices_mm), "faces": len(faces),
                      "size_xyz_mm": (maximum - minimum).tolist(), "uv_error": uv_error}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
