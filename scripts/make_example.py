"""Extract an existing observation for the standalone inference example."""
from pathlib import Path
import json,av,numpy as np,pyarrow.parquet as pq
root=Path('data/robotwin_clean_xera_v21');episode=0
row=pq.read_table(root/'data/chunk-000/episode_000000.parquet').slice(0,1).to_pylist()[0]
task_index=row['task_index'];tasks={r['task_index']:r['task'] for r in map(json.loads,(root/'meta/tasks.jsonl').open())}
statekey=next(k for k in row if k.startswith('observation.state'))
payload={'state':np.asarray(row[statekey],np.float32),'instruction':np.array(tasks[task_index])}
for key in ['cam_high','cam_left_wrist','cam_right_wrist']:
 with av.open(str(root/f'videos/chunk-000/observation.images.{key}/episode_000000.mp4')) as c:
  payload[key]=next(c.decode(video=0)).to_ndarray(format='rgb24')
Path('examples').mkdir(exist_ok=True);np.savez_compressed('examples/observation.npz',**payload)
print('Wrote examples/observation.npz')
