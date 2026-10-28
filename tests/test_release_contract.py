from pathlib import Path
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from scripts.infer import predict

def test_release_config_matches_checkpoint_recipe():
 c=OmegaConf.load('configs/release.yaml')
 assert c.model._target_=="e05.create_e05"
 assert c.batch_size==8 and c.gradient_accumulation_steps==1 and c.num_epochs==70
 assert c.stop_after_steps is None and c.max_steps is None
 assert c.optimizer_grouping=='legacy' and c.model.reconstruction_target=='future'
 assert c.data.train.reconstruction_target=='future'
 assert c.data.train.processor.vggt_augmentation.enabled
 assert c.model.reconstruction_loss_config.backend=='official' and c.model.lambda_vggt_recon==0.01
 assert not c.wandb.enabled
 assert '/meta_eon_cfs/' not in OmegaConf.to_yaml(c)

@pytest.mark.parametrize('geometry', [False, True])
def test_inference_passes_language_mask_and_current_state_without_gt(geometry):
 observed={}
 class Model:
  def infer_action(self,**kwargs):
   observed.update(kwargs)
   action=torch.ones(1,32,14)
   if kwargs['return_reconstruction']:
    return {'action':action,'reconstruction':{'future':{'depth':torch.ones(1,3,2,2,1),'scale_convention':'scene_normalized'}}}
   return action
 class Policy:
  model=Model()
  def _get_text_context(self,prompt):return torch.zeros(128,4096),torch.ones(128,dtype=torch.bool),torch.arange(128)<5
  def _build_robotwin_vggt_tensor(self,obs):return torch.zeros(1,3,3,224,288)
  def _normalize_state(self,state):return torch.tensor(state)[None]
  def _denormalize_action(self,action):return action.numpy()*2
 payload={k:np.zeros((240,320,3),dtype=np.uint8) for k in ['cam_high','cam_left_wrist','cam_right_wrist']}
 payload.update(state=np.arange(14,dtype=np.float32),instruction=np.array('pick the bottle'))
 out=predict(Policy(),payload,geometry)
 if geometry:
  assert out['future_scale_convention'].item()=='scene_normalized'
  assert out['future_depth'].shape==(1,3,2,2,1)
 assert out['action'].shape==(32,14) and (out['action']==2).all()
 assert observed['language_context_mask'].sum()==5 and observed['context_mask'].all()
 assert observed['proprio'].shape==(1,14)
 assert observed['action_horizon']==32 and observed['num_inference_steps']==10


def test_robotwin_runtime_task_configs_are_included():
 import subprocess
 import yaml
 root=Path('third_party/RoboTwin/task_config')
 for name in ('_camera_config.yml','_embodiment_config.yml','_eval_step_limit.yml','demo_clean.yml','demo_randomized.yml'):
  path=root/name
  assert isinstance(yaml.safe_load(path.read_text()),dict),path
  result=subprocess.run(['git','check-ignore',str(path)],capture_output=True)
  assert result.returncode==1,(path,result.stdout,result.stderr)
 for name in ('demo_clean.yml','demo_randomized.yml'):
  assert yaml.safe_load((root/name).read_text())['embodiment']==['aloha-agilex']
