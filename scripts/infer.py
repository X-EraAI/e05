"""RGB + instruction + current state inference without importing the simulator."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

def load_policy(config,checkpoint,device='cuda',online=True):
 cfg=OmegaConf.load(config)
 stats=Path(checkpoint).parent/'dataset_stats.json'
 if not stats.is_file():stats=Path(cfg.data.train.pretrained_norm_stats)
 return WorldActionRobotWinPolicy(
  model_cfg=cfg.model,processor_cfg=cfg.data.train.processor,checkpoint_path=str(checkpoint),
  dataset_stats_path=stats,device=device,model_dtype=torch.bfloat16,
  action_horizon=32,replan_steps=24,num_inference_steps=10,sigma_shift=None,seed=42,
  text_cfg_scale=1.0,negative_prompt='',rand_device='cpu',tiled=False,timing_enabled=True,
  num_video_frames=9,text_embedding_cache_dir=cfg.data.train.text_embedding_cache_dir,context_len=128,
  online_text_embeddings=online,write_text_embedding_cache=False)

def predict(policy,payload,reconstruction=False):
 mapping={'cam_high':'head_camera','cam_left_wrist':'left_camera','cam_right_wrist':'right_camera'}
 state=np.asarray(payload['state'],dtype=np.float32)
 if state.shape!=(14,) or not np.isfinite(state).all():raise ValueError('state must be finite float32[14]')
 obs={'joint_action':{'vector':state},'observation':{}}
 for key,cam in mapping.items():
  rgb=np.asarray(payload[key])
  if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3:raise ValueError(f'{key}: expected uint8 RGB[H,W,3]')
  obs['observation'][cam]={'rgb':rgb}
 context,mask,language_mask=policy._get_text_context(DEFAULT_PROMPT.format(task=str(payload['instruction'].item())))
 with torch.inference_mode():
  out=policy.model.infer_action(vggt_images=policy._build_robotwin_vggt_tensor(obs),context=context,
   context_mask=mask,language_context_mask=language_mask,proprio=policy._normalize_state(state),
   action_horizon=32,num_inference_steps=10,seed=42,rand_device='cpu',return_reconstruction=reconstruction)
 norm=out['action'] if isinstance(out,dict) else out
 action=policy._denormalize_action(norm)[0]
 if action.shape!=(32,14) or not np.isfinite(action).all():raise ValueError('Invalid predicted actions')
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
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
 p.add_argument('--config',default='configs/release.yaml');p.add_argument('--checkpoint',default='checkpoints/release/step_109000.pt')
 p.add_argument('--device',default='cuda');p.add_argument('--reconstruction',action='store_true');p.add_argument('--cached-text-only',action='store_true')
 a=p.parse_args();start=time.time();policy=load_policy(a.config,a.checkpoint,a.device,not a.cached_text_only)
 with np.load(a.input,allow_pickle=False) as payload:result=predict(policy,payload,a.reconstruction)
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**result)
 report={'passed':True,'model_name':policy.model.MODEL_NAME,'model_class':type(policy.model).__module__+'.'+type(policy.model).__name__,'checkpoint':str(a.checkpoint),'shapes':{k:list(v.shape) for k,v in result.items()},'elapsed_seconds':time.time()-start,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
 a.output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
