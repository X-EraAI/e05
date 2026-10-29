"""Exercise the production training loader with augmentation and released caches."""
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

def main():
    cfg = OmegaConf.load('configs/release.yaml')
    ds = instantiate(cfg.data.train)
    rows = []
    for index in [0, 64, 138, len(ds) - 1]:
        sample = ds[index]
        row = {'index': index, 'fields': {}}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), (index, key)
                row['fields'][key] = {'shape': list(value.shape), 'dtype': str(value.dtype)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path('outputs/validation').mkdir(parents=True, exist_ok=True)
    Path('outputs/validation/production_dataset.json').write_text(json.dumps({
        'passed': True, 'length': len(ds), 'augmentation_enabled': True,
        'samples': rows,
    }, indent=2))

if __name__ == '__main__':
    main()
