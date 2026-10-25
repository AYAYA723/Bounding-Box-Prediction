import ast
import hashlib
import json
import subprocess
from pathlib import Path

import torch


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / ".cache/boxdreamer-source"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source).decode().strip()
    relative = "src/models/utils/box_utils.py"
    module = ast.parse((source / relative).read_text(encoding="utf-8"))
    definition = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                      and node.name == "recover_bb8_corners")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), relative, "exec"), scope)
    torch.manual_seed(42)
    feature = torch.randn(2, 1, 12, 16, 8, requires_grad=True)
    normalized, pixels = scope["recover_bb8_corners"](feature, "heatmap")
    loss_config = (source / "configs/model/loss/default.yaml").read_text()
    active = "\n".join(line for line in loss_config.splitlines() if not line.lstrip().startswith("#"))
    files = ["configs/model/loss/default.yaml", relative,
             "src/datasets/utils/base/bbox_utils.py", "src/models/modules/backbone/betr.py"]
    report = {
        "commit": commit, "repository": "https://github.com/zju3dv/BoxDreamer",
        "default_fine_loss_enabled": "regression_boxes" in active,
        "official_coordinate_method": "mean pixel location of top-20 heatmap entries",
        "official_coordinates_require_grad": pixels.requires_grad,
        "official_normalized_coordinates_require_grad": normalized.requires_grad,
        "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files},
        "selected_target": "paper coarse + 2*fine; user explicitly chose differentiable completion",
    }
    output = root / "results/paper_supervision_implementation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
