"""RGB + instruction + current state inference without importing the simulator.

Camera keys, their order, the state size and the action size come from the
training configuration: `--config` accepts a full configuration (configs/release.yaml
or a run's saved config.yaml) or a task file such as a filled
configs/task/real_robot_template.yaml, which is merged onto configs/release.yaml.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from e05.checkpoint_compat import normalize_config
from e05.datasets.lerobot.processors.e05_processor import prepare_vggt_images
from experiments.robotwin.e05_policy.deploy_policy import WorldActionRobotWinPolicy
from e05.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
ROOT=Path(__file__).resolve().parents[1]

def load_config(config):
 cfg=OmegaConf.load(config)
 # A task file has no model target of its own; compose it like `+task=...` does.
 if OmegaConf.select(cfg,'model._target_') is None:cfg=OmegaConf.merge(OmegaConf.load(ROOT/'configs/release.yaml'),cfg)
 return normalize_config(cfg)

def resolve_stats(cfg,config,checkpoint,stats=None):
 if stats:return Path(stats)
 for path in (Path(checkpoint).parent/'dataset_stats.json',Path(config).parent/'dataset_stats.json'):
  if path.is_file():return path
 if not cfg.data.train.pretrained_norm_stats:raise FileNotFoundError('No dataset_stats.json beside the checkpoint or config; pass --stats')
 return Path(cfg.data.train.pretrained_norm_stats)

def load_policy(config,checkpoint,device='cuda',online=True,stats=None):
 if not Path(checkpoint).is_file():raise FileNotFoundError(f'Inference checkpoint not found: {checkpoint}')
 cfg=load_config(config)
 return WorldActionRobotWinPolicy(
  model_cfg=cfg.model,processor_cfg=cfg.data.train.processor,checkpoint_path=str(checkpoint),
  dataset_stats_path=resolve_stats(cfg,config,checkpoint,stats),device=device,model_dtype=torch.bfloat16,
  action_horizon=int(cfg.data.train.num_frames)-1,replan_steps=24,num_inference_steps=10,sigma_shift=None,seed=42,
  text_cfg_scale=1.0,negative_prompt='',rand_device='cpu',tiled=False,timing_enabled=True,
  num_video_frames=9,text_embedding_cache_dir=cfg.data.train.text_embedding_cache_dir,context_len=128,
  online_text_embeddings=online,write_text_embedding_cache=False)

def predict(policy,payload,reconstruction=False):
 meta=policy.processor.shape_meta;keys=list(policy.processor.vggt_camera_keys)
 state_dim=int(meta['state'][0]['raw_shape']);action_dim=int(meta['action'][0]['shape'])
 state=np.asarray(payload['state'],dtype=np.float32)
 if state.shape!=(state_dim,) or not np.isfinite(state).all():raise ValueError(f'state must be finite float32[{state_dim}]')
 images={}
 for key in keys:
  if key not in payload:raise KeyError(f'Observation is missing camera {key!r}; expected {keys}')
  rgb=np.asarray(payload[key])
  if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3:raise ValueError(f'{key}: expected uint8 RGB[H,W,3]')
  images[key]=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).unsqueeze(0)
 vggt_images=prepare_vggt_images(images,camera_keys=keys,image_size=int(policy.processor.vggt_image_size)).unsqueeze(0).contiguous()
 context,mask,language_mask=policy._get_text_context(DEFAULT_PROMPT.format(task=str(payload['instruction'].item())))
 with torch.inference_mode():
  out=policy.model.infer_action(vggt_images=vggt_images,context=context,
   context_mask=mask,language_context_mask=language_mask,proprio=policy._normalize_state(state),
   action_horizon=policy.action_horizon,num_inference_steps=10,seed=42,rand_device='cpu',return_reconstruction=reconstruction)
 norm=out['action'] if isinstance(out,dict) else out
 action=policy._denormalize_action(norm)[0]
 if action.shape!=(policy.action_horizon,action_dim) or not np.isfinite(action).all():raise ValueError('Invalid predicted actions')
 result={'action':action,'normalized_action':norm[0].numpy()}
 if isinstance(out,dict):
  for target,geometry in out['reconstruction'].items():
   for k,v in geometry.items():
    if isinstance(v,torch.Tensor):
     if not torch.isfinite(v).all():raise ValueError(f'Non-finite geometry: {k}')
     result[f'{target}_{k}']=v.numpy()
    elif isinstance(v,(str,bool,int,float)):
     result[f'{target}_{k}']=np.asarray(v)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
 p.add_argument('--config',default='configs/release.yaml');p.add_argument('--checkpoint',default='checkpoints/release/step_109000.pt')
 p.add_argument('--stats',default=None,help='dataset_stats.json; default: beside the checkpoint, beside --config, then data.train.pretrained_norm_stats')
 p.add_argument('--device',default='cuda');p.add_argument('--reconstruction',action='store_true');p.add_argument('--cached-text-only',action='store_true')
 a=p.parse_args();start=time.time();policy=load_policy(a.config,a.checkpoint,a.device,not a.cached_text_only,a.stats)
 with np.load(a.input,allow_pickle=False) as payload:result=predict(policy,payload,a.reconstruction)
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**result)
 report={'passed':True,'model_name':policy.model.MODEL_NAME,'model_class':type(policy.model).__module__+'.'+type(policy.model).__name__,'checkpoint':str(a.checkpoint),'shapes':{k:list(v.shape) for k,v in result.items()},'elapsed_seconds':time.time()-start,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
 a.output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
