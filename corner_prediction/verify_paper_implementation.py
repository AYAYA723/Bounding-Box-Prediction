import csv
import json
from pathlib import Path

import torch
from torch import optim

from dataset import CornerDataset
from model import CornerPredictor
from supervision import BoxDreamerLoss
from transformer_model import TransformerCornerPredictor


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    output = root / "results/paper_supervision_implementation"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dataset = CornerDataset(root / "datasets/action4/train_pbr/000000")
    image, maps, target = dataset[0]
    x, y, target = image[None].to(device), maps[None].to(device), target[None].to(device)
    criterion = BoxDreamerLoss()
    report = {"loss": "coarse + 2*fine", "decoder": "logit soft-argmax, temperature=1",
              "normalization": "2*xy/[W,H]-1", "architectures": {}}
    for name, model_class in (("resnet", CornerPredictor), ("vit", TransformerCornerPredictor)):
        torch.manual_seed(42)
        model = model_class().to(device)
        pred, corners = model(x)
        loss, coarse, fine = criterion(pred, corners, y, target)
        encoder_weight = model.encoder[0].weight if name == "resnet" else model.patch_embed.weight
        gradient = torch.autograd.grad(fine, encoder_weight, retain_graph=True)[0]
        assert gradient.norm() > 0 and torch.isfinite(gradient).all()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        report["architectures"][name] = {
            "heatmap_shape": list(pred.shape), "corner_shape": list(corners.shape),
            "fine_only_encoder_gradient_norm": gradient.norm().item(),
            "all_parameter_gradients_finite": True,
            "initial_total": loss.item(), "initial_coarse": coarse.item(), "initial_fine": fine.item()}
        del model, pred, corners, loss, coarse, fine, gradient

    torch.manual_seed(42)
    model = CornerPredictor().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    history = []
    for step in range(201):
        pred, corners = model(x)
        loss, coarse, fine = criterion(pred, corners, y, target)
        error = (corners - target).norm(dim=-1).mean().item()
        history.append([step, loss.item(), coarse.item(), fine.item(), error])
        if step % 20 == 0:
            print(f"Single image step {step}: total={loss.item():.6f}, coarse={coarse.item():.6f}, "
                  f"fine={fine.item():.6f}, corner={error:.3f}px", flush=True)
        if step < 200:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        pred, corners = model(x)
        loss, coarse, fine = criterion(pred, corners, y, target)
        eval_error = (corners - target).norm(dim=-1).mean().item()
    report["single_image_check"] = {
        "image": "000000.png", "steps": 200, "learning_rate": 3e-4,
        "initial_train_error_px": history[0][-1], "final_train_error_px": history[-1][-1],
        "final_eval_error_px": eval_error, "final_eval_total": loss.item(),
        "final_eval_coarse": coarse.item(), "final_eval_fine": fine.item(),
        "note": "Implementation check on one training image; not validation or video accuracy."}
    with (output / "single_image_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "total_loss", "coarse_loss", "fine_loss", "train_corner_error_px"])
        writer.writerows(history)
    torch.save(model.state_dict(), output / "single_image_check.pt")
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
