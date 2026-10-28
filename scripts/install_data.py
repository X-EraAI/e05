"""Install a local E05 processed-data release. No raw-data generation is performed."""
import argparse,json,hashlib,shutil,zipfile
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('release',type=Path);p.add_argument('--output',type=Path,default=Path('data'));a=p.parse_args()
 manifest=json.loads((a.release/'manifest.json').read_text());a.output.mkdir(parents=True,exist_ok=True)
 for item in manifest['files']:
  src=a.release/item['file'];assert sha(src)==item['sha256'],src
  if item['mode']=='extract':
   with zipfile.ZipFile(src) as z:
    for member in z.infolist():
     target=(a.output/member.filename).resolve()
     if not target.is_relative_to(a.output.resolve()):raise ValueError('Unsafe archive path')
    z.extractall(a.output)
  else:
   dst=a.output/item['target'];assert dst.resolve().is_relative_to(a.output.resolve());dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 print('Installed processed data. Run scripts/validate_data.py --full.')
if __name__=='__main__':main()
