"""Validate the prepared PLY with the checkout's own BOP reader; compute diameter."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--repo", type=Path, required=True)
args = parser.parse_args()
dataset = args.dataset.resolve()
archive = args.repo / "bop_toolkit.zip"
with zipfile.ZipFile(archive) as zipped:
    source = zipped.read("bop_toolkit/bop_toolkit_lib/inout.py").decode("utf-8")
parsed = ast.parse(source)
function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "load_ply")
namespace = {"np": np, "struct": struct}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(archive) + "/inout.py", "exec"), namespace)
ply = dataset / "models/obj_000001.ply"
model = namespace["load_ply"](str(ply))
vertices, faces = model["pts"], model["faces"].astype(np.int64)
assert model["texture_file"] == "obj_000001.png"
assert np.isfinite(vertices).all() and np.isfinite(model["normals"]).all()
assert faces.min() >= 0 and faces.max() < len(vertices)
assert model["texture_uv"].shape == (len(vertices), 2)
assert np.max(np.abs(np.linalg.norm(model["normals"], axis=1) - 1)) < 1e-5
with Image.open(dataset / "models" / model["texture_file"]) as image:
    texture_size = image.size
    image.verify()
assert texture_size == (4096, 4096)
minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
assert np.max(np.abs(minimum + maximum)) < 1e-5
assert abs((maximum - minimum)[0] - 70.5) < 1e-5
print("BOP load_ply, texture, UVs, normals, scale and center: PASS", flush=True)

# The farthest pair of a finite point set lies on its convex hull. Chunked cdist
# computes this diameter exactly for the exported mesh without an NxN allocation.
hull = ConvexHull(vertices)
extremes = vertices[hull.vertices]
largest = 0.0
for start in range(0, len(extremes), 256):
    largest = max(largest, float(cdist(extremes[start:start + 256], extremes, metric="sqeuclidean").max()))
diameter = float(np.sqrt(largest))
info = {"diameter": diameter}
for axis, i in zip("xyz", range(3)):
    info.update({"min_" + axis: float(minimum[i]), "max_" + axis: float(maximum[i]),
                 "size_" + axis: float(maximum[i] - minimum[i])})
(dataset / "models/models_info.json").write_text(json.dumps({"1": info}, indent=2), encoding="utf-8")
corners = np.array(json.loads((dataset / "keypoints_3d.json").read_text())["corners"])
assert corners.shape == (8, 3)
assert np.allclose(corners.min(axis=0), minimum, atol=1e-5)
assert np.allclose(corners.max(axis=0), maximum, atol=1e-5)
manifest_path = dataset / "metadata/preparation.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["status"] = "geometry_metadata_validated_pending_preview"
manifest["validation"].update({"hccepose_bop_reader": "PASS", "png_integrity": "PASS", "normals": "PASS",
                               "bbox_keypoints_consistent": "PASS", "aabb_center_mm": ((minimum + maximum) / 2).tolist(),
                               "diameter_mm": diameter, "diameter_method": "exact farthest pair of convex-hull vertices, chunked cdist",
                               "convex_hull_vertices": len(extremes), "bop_toolkit_zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
manifest["output_hashes"] = {str(path.relative_to(dataset)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in [ply, dataset / "models/obj_000001.png", dataset / "models/models_info.json",
                                         dataset / "camera.json", dataset / "keypoints_3d.json"]}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"vertices": len(vertices), "triangles": len(faces), "diameter_mm": diameter,
                  "size_xyz_mm": (maximum - minimum).tolist(), "hull_vertices": len(extremes)}), flush=True)
