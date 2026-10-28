"""Build local code/model/data artifacts without uploading them."""
import argparse,json,hashlib,zipfile,tarfile,shutil,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('dist'));mode=p.add_mutually_exclusive_group();mode.add_argument('--code-only',action='store_true');mode.add_argument('--code-and-model',action='store_true');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 code=out/'E05-code.tar.gz'
 with tarfile.open(code,'w:gz',compresslevel=1) as tar:
  for name in ['src','scripts','configs','tests','docs','examples','experiments','third_party','artifacts','README.md','LICENSE','THIRD_PARTY_NOTICES.md','pyproject.toml','requirements.txt','requirements.lock.txt','environment.conda.lock.txt','.gitignore']:
   pth=ROOT/name
   if not pth.exists():continue
   def filt(info):
    parts=Path(info.name).parts
    if any(x in parts for x in ['__pycache__','.git','.pytest_cache']):return None
    if info.name.startswith('E05/third_party/RoboTwin/assets'):return None
    if info.name.endswith(('.pyc','.so','.log')):return None
    info.uid=info.gid=0;info.uname=info.gname='';return info
   tar.add(pth,arcname='E05/'+name,filter=filt)
 if a.code_only:
  sums=out/'SHA256SUMS'
  lines=sums.read_text().splitlines() if sums.exists() else []
  lines=[line for line in lines if not line.endswith('  '+code.name)]
  sums.write_text(f'{sha(code)}  {code.name}\n'+'\n'.join(lines)+'\n')
  print('Code archive refreshed:',code);return
 model=out/'E05-model-109000.tar'
 with tarfile.open(model,'w') as tar:
  for pth in sorted((ROOT/'checkpoints/release').iterdir()):tar.add(pth,arcname='checkpoints/release/'+pth.name)
 if a.code_and_model:
  sums=out/'SHA256SUMS'
  lines=sums.read_text().splitlines() if sums.exists() else []
  lines=[line for line in lines if not any(line.endswith('  '+p.name) for p in (code,model))]
  sums.write_text(f'{sha(code)}  {code.name}\n{sha(model)}  {model.name}\n'+'\n'.join(lines)+'\n')
  print('Code and model metadata refreshed:',out);return
 dataout=out/'data';dataout.mkdir(exist_ok=True);entries=[]
 for folder in ['reconstruction_cache','text_embeds_cache/robotwin']:
  if not (ROOT/'data'/folder/'manifest.json').exists():raise RuntimeError(f'Incomplete cache release: {folder}')
  manifest=json.loads((ROOT/'data'/folder/'manifest.json').read_text())
  for item in manifest['shards']:
   archive=ROOT/item['path']
   if not archive.is_file() or archive.stat().st_size!=item['size']:raise RuntimeError(f'Missing/incomplete cache shard: {archive}')
 ds=ROOT/'data/robotwin_clean_xera_v21';eps=[json.loads(x) for x in (ds/'meta/episodes.jsonl').open()]
 def task_pack(task):
  name=f'{task}.zip';target=dataout/name
  if target.exists():
   try:
    with zipfile.ZipFile(target) as z:
     if len(z.infolist())!=sum(e['xera_source']['task']==task for e in eps)*4:raise zipfile.BadZipFile('Incomplete task archive')
   except zipfile.BadZipFile:target.unlink()
  if not target.exists():
   with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
    for e in eps:
     if e['xera_source']['task']!=task:continue
     i=e['episode_index'];c=i//1000
     paths=[ds/f'data/chunk-{c:03d}/episode_{i:06d}.parquet']+[ds/f'videos/chunk-{c:03d}/observation.images.{cam}/episode_{i:06d}.mp4' for cam in ['cam_high','cam_left_wrist','cam_right_wrist']]
     for src in paths:z.write(src,str(src.relative_to(ROOT/'data')))
  return {'file':name,'mode':'extract','size':target.stat().st_size,'sha256':sha(target)}
 with ThreadPoolExecutor(max_workers=8) as pool:entries.extend(pool.map(task_pack,sorted({e['xera_source']['task'] for e in eps})))
 meta=dataout/'metadata.zip'
 with zipfile.ZipFile(meta,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for src in (ds/'meta').iterdir():
   if src.is_file():z.write(src,str(src.relative_to(ROOT/'data')))
  z.write(ROOT/'data/dataset_stats.json','dataset_stats.json')
 entries.append({'file':meta.name,'mode':'extract','size':meta.stat().st_size,'sha256':sha(meta)})
 # Cache shards already are independent distributable archives. Copy them unchanged.
 for folder in ['reconstruction_cache','text_embeds_cache/robotwin']:
  for src in sorted((ROOT/'data'/folder).rglob('*')):
   if not src.is_file() or (src.suffix not in ['.zip','.json']):continue
   rel=src.relative_to(ROOT/'data');dest=dataout/rel;dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():
    if src.suffix=='.zip':
     # Atomic swap keeps live dataset readers valid. The temporary
     # hard link exists only between two files inside this new project.
     os.link(src,dest)
     link=src.with_name(src.name+'.release-link')
     link.symlink_to(os.path.relpath(dest,src.parent))
     os.replace(link,src)
    else:shutil.copyfile(src,dest)
   checksum=json.loads(src.with_suffix('.json').read_text())['sha256'] if src.suffix=='.zip' else sha(dest)
   entries.append({'file':str(rel),'mode':'copy','target':str(rel),'size':dest.stat().st_size,'sha256':checksum})
 (dataout/'manifest.json').write_text(json.dumps({'version':'e05-109k-v1','files':entries},indent=2))
 sums=f'{sha(code)}  {code.name}\n{sha(model)}  {model.name}\n{sha(dataout/"manifest.json")}  data/manifest.json\n'
 asset=out/'E05-simulator-assets.zip'
 if asset.exists():sums+=f'{sha(asset)}  {asset.name}\n'
 (out/'SHA256SUMS').write_text(sums)
 print('Release built:',out)
if __name__=='__main__':main()
