import json,sys,re
from pathlib import Path
from omegaconf import OmegaConf
p=Path(sys.argv[1]); d=json.loads((p/'summary.json').read_text())
rows=d.get('per_task',d.get('tasks',[]))
assert len(rows)==8,(len(rows),d.keys())
for row in rows:
 for phase in ('clean','random'):
  assert row[f'{phase}_success_rate'] is not None,row
expected={'adjust_bottle','beat_block_hammer','handover_block','open_microwave','pick_dual_bottles','place_bread_basket','stack_blocks_two','turn_switch'}
assert {row['task_name'] for row in rows}==expected
rollouts=[]
for task in sorted(expected):
 for phase in ('clean','random'):
  config=OmegaConf.load(p/f'eval_config_{task}_demo_{"clean" if phase=="clean" else "randomized"}.yaml')
  assert config.model._target_=='e05.create_e05'
  assert config.EVALUATION.policy_name=='e05_policy'
  assert Path(config.ckpt).name=='step_109000.pt'
  assert config.seed==42 and config.EVALUATION.eval_num_episodes==1
  assert config.EVALUATION.action_horizon==32 and config.EVALUATION.replan_steps==24 and config.EVALUATION.num_inference_steps==10
  assert config.EVALUATION.instruction_type==('seen' if phase=='clean' else 'unseen')
  matches=[]
  for log in p.glob(f'eval_{task}_*.log'):
   text=re.sub(r'\x1b\[[0-9;]*m','',log.read_text())
   assert 'Traceback (most recent call last)' not in text,log
   assert 'OIDN Error:' not in text, f'Rendering denoiser error: {log}'
   if f'_result_{phase}.txt' not in text:continue
   counts=re.findall(r'Success rate: (\d+)/(\d+)',text)
   if counts:matches.append((log,counts[-1]))
  assert len(matches)==1,(task,phase,matches)
  log,(success,total)=matches[0]
  assert int(total)==1 and int(success) in (0,1),(task,phase,success,total)
  rollouts.append({'task':task,'phase':phase,'completed':int(total),'successes':int(success),'log':log.name})
failed=p/'failed_tasks.txt'
assert not failed.exists() or not failed.read_text().strip(),failed
report={'passed':True,'model_name':'e05','model_factory':'e05.create_e05','tasks':8,'episodes':16,'summary':d,'rollouts':rollouts,'success_rate_is_not_a_pass_threshold':True}
(p/'smoke_report.json').write_text(json.dumps(report,indent=2))
print('PASS: 8 tasks / 16 complete rollout results')
