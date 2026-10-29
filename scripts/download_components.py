"""Download exactly the component files recorded in the release manifest."""
import argparse,json,hashlib
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('checkpoints'));a=p.parse_args()
 items=json.loads(Path('configs/resources/component_manifest.json').read_text())['files']
 for item in items:
  target=a.root/item['path']
  if target.exists() and sha(target)==item['sha256']:continue
  if item['modelscope_repo'] is None:continue
  from modelscope import snapshot_download
  import shutil
  repo=item['modelscope_repo'];downloaded=Path(snapshot_download(repo,revision=item['revision'],allow_file_pattern=[item['file']]))/item['file']
  if sha(downloaded)!=item['sha256']:raise ValueError(f'Upstream bytes differ from release: {repo}/{item["file"]}')
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(downloaded,target)
 # The aggregator is a tensor-identical extraction, not another learned initialization.
 import torch
 full=a.root/'VGGT-Omega/vggt_omega_1b_256_text.pt';dest=a.root/'VGGT-Omega/vggt_omega_1b_256_aggregator.pt'
 if not dest.exists():
  from e05.models.wan22.vggt_mot import export_aggregator_checkpoint
  export_aggregator_checkpoint(full,dest)
 source=torch.load(full,map_location='cpu',mmap=True,weights_only=True)
 for key in ('state_dict','model'):
  if key in source and isinstance(source[key],dict):source=source[key]
 extracted={k[len('aggregator.'):]:v for k,v in source.items() if k.startswith('aggregator.')}
 actual=torch.load(dest,map_location='cpu',mmap=True,weights_only=True)
 assert extracted.keys()==actual.keys()
 assert all(torch.equal(extracted[k],actual[k]) for k in extracted),'Aggregator does not match public initialization'
 print('Components verified; aggregator tensor contents match the public checkpoint.')
if __name__=='__main__':main()
