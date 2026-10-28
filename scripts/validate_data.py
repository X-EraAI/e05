"""Validate an installed E05 processed dataset; never generates training labels."""
import argparse,json,hashlib,zipfile,math
from pathlib import Path
import torch
from concurrent.futures import ThreadPoolExecutor
from fastwam.datasets.cache_io import load_cache

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=Path('data'));p.add_argument('--full',action='store_true');p.add_argument('--report',type=Path,default=Path('artifacts/data_validation.json'));a=p.parse_args()
 root=a.data_root;ds=root/'robotwin_clean_xera_v21';info=json.loads((ds/'meta/info.json').read_text())
 eps=[json.loads(x) for x in (ds/'meta/episodes.jsonl').open()]
 assert info['total_episodes']==len(eps)==2500
 assert sum(e['length'] for e in eps)==info['total_frames']==549787
 assert len({e['xera_source']['task'] for e in eps})==50
 for e in eps:
  idx=e['episode_index'];chunk=idx//1000
  assert (ds/f'data/chunk-{chunk:03d}/episode_{idx:06d}.parquet').is_file()
  for cam in ['cam_high','cam_left_wrist','cam_right_wrist']:
   assert (ds/f'videos/chunk-{chunk:03d}/observation.images.{cam}/episode_{idx:06d}.mp4').is_file()
 tasks=[json.loads(x)['task'] for x in (ds/'meta/tasks.jsonl').open()]
 expected_text={hashlib.sha256(("A video recorded from a robot's point of view executing the following instruction: "+t).encode()).hexdigest()+'.t5_len128.wan22ti2v5b.pt' for t in tasks}
 counts={}
 for kind,folder,expected in [('geometry',root/'reconstruction_cache',{f'{i:08d}.pt' for i in range(info['total_frames'])}),('text',root/'text_embeds_cache/robotwin',expected_text)]:
  manifest=json.loads((folder/'manifest.json').read_text());actual=set()
  if a.full:
   def check_hash(item):
    path=folder/'shards'/Path(item['path']).name
    if digest(path)!=item['sha256']:raise ValueError(f'Checksum mismatch: {path}')
    return True
   with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(check_hash,manifest['shards']))
  for shard in manifest['shards']:
   path=folder/'shards'/Path(shard['path']).name
   assert path.stat().st_size==shard['size'],path
   with zipfile.ZipFile(path) as z:names=z.namelist()
   assert not actual.intersection(names),path
   actual.update(names)
   metadata=json.loads(path.with_suffix('.json').read_text())
   assert set(names)=={x['name'] for x in metadata['entries']},path
  assert actual==expected,(kind,len(actual),len(expected),list(expected-actual)[:5])
  counts[kind]=len(actual)
 sample_indices=[0,64,eps[0]['length']-1,info['total_frames']//2,info['total_frames']-1]
 for i in sample_indices:
  v=load_cache(root/'reconstruction_cache'/f'{i:08d}.pt')
  assert v['depth'].shape==(3,240,320) and v['intrinsics'].shape==(3,3,3) and v['extrinsics'].shape==(3,3,4)
  assert torch.isfinite(v['intrinsics']).all() and torch.isfinite(v['extrinsics']).all()
  valid=v.get('valid',v['depth']>0).bool();assert torch.isfinite(v['depth'][valid]).all() and (v['depth'][valid]>0).all()
 for name in sorted(expected_text)[::max(1,len(expected_text)//10)]:
  v=load_cache(root/'text_embeds_cache/robotwin'/name)
  assert v['context'].shape==(128,4096) and v['mask'].shape==(128,)
  assert torch.isfinite(v['context']).all()
 report={'passed':True,'full_archive_hash_check':a.full,'episodes':len(eps),'frames':info['total_frames'],'robot_tasks':50,'language_entries':len(tasks),'fps':info['fps'],'cache_counts':counts,'sample_indices':sample_indices}
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
