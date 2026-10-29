import json,os,sys,importlib,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
(root/"audit").mkdir(exist_ok=True)
(root/"outputs/validation").mkdir(parents=True,exist_ok=True)
report={'python':sys.executable,'prefix':sys.prefix,'modules':{},'cache_paths':{}}
assert Path(sys.prefix).resolve().is_relative_to(root),sys.prefix
for name in ['torch','e05','vggt_omega','curobo','sapien','mplib','av','datasets','deepspeed','transformers']:
 m=importlib.import_module(name);p=Path(m.__file__).resolve();report['modules'][name]={'path':str(p),'version':getattr(m,'__version__',None)}
 assert p.is_relative_to(root),(name,p)
for key in ['HF_HOME','HF_DATASETS_CACHE','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','WARP_CACHE_PATH','TMPDIR']:
 p=Path(os.environ[key]).resolve();assert p.is_relative_to(root);report['cache_paths'][key]=str(p)
import torch
report['gpu_count']=torch.cuda.device_count();report['gpus']=[{'name':torch.cuda.get_device_name(i),'capability':torch.cuda.get_device_capability(i),'free_bytes':torch.cuda.mem_get_info(i)[0]} for i in range(torch.cuda.device_count())]
assert report['gpu_count']>=8,'Eight visible GPUs are required for the planned smoke'
report['passed']=True
(root/'audit/environment_report.full.json').write_text(json.dumps(report,indent=2))
public=json.loads(json.dumps(report).replace(str(root),'${E05_ROOT}'))
(root/'outputs/validation/environment_report.json').write_text(json.dumps(public,indent=2))
print(json.dumps(report,indent=2))
