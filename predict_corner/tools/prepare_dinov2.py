import hashlib
import json
from pathlib import Path

import torch


def main():
    root = Path(__file__).resolve().parents[2]
    cache = root / '.cache/torch'
    torch.hub.set_dir(str(cache))
    model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitb14_reg',
                           pretrained=True, trust_repo=True, skip_validation=True)
    weights = cache / 'checkpoints/dinov2_vitb14_reg4_pretrain.pth'
    repository = cache / 'facebookresearch_dinov2_main'
    report = {'encoder': 'dinov2_vitb14_reg', 'source': 'https://github.com/facebookresearch/dinov2',
              'parameters': sum(p.numel() for p in model.parameters()), 'weights': str(weights),
              'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
              'source_sha256': {name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
                                for name in ['dinov2/models/vision_transformer.py', 'dinov2/layers/attention.py',
                                             'dinov2/hub/backbones.py']}}
    (cache / 'dinov2_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
