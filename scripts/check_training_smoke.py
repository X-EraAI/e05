import json,sys,math,re
from pathlib import Path
from omegaconf import OmegaConf
p=Path(sys.argv[1]);reports=[]
cfg=OmegaConf.load(p/'config.yaml');batch=int(cfg.batch_size)
accum=int(cfg.gradient_accumulation_steps)
assert batch in (16,8,4,2) and accum in (1,2,4)
assert cfg.stop_after_steps==4 and cfg.num_epochs==70
assert cfg.model._target_=='e05.create_e05'
assert cfg.resume is None and cfg.model.pretrained_checkpoint_path is None
assert cfg.model.action_dit_pretrained_path is None and cfg.model.skip_action_dit_load_from_pretrain
assert cfg.model.train_vggt and not cfg.model.skip_vggt_load_from_pretrain
assert cfg.optimizer_grouping=='legacy' and cfg.mixed_precision=='bf16'
assert cfg.data.train.processor.vggt_augmentation.enabled
assert cfg.model.lambda_vggt_recon==0.01 and cfg.model.reconstruction_loss_config.backend=='official'
for rank in range(8):
 rows=[json.loads(x) for x in (p/f'audit/rank_{rank}.jsonl').read_text().splitlines()]
 assert [x['step'] for x in rows]==[1,2,3,4],(rank,rows)
 assert [x['microbatches_completed'] for x in rows]==[accum*i for i in (1,2,3,4)],rank
 assert all(x['gradient_accumulation_steps']==accum for x in rows),rank
 assert all(x['model_name']=='e05' and x['model_class'].endswith('.e05.E05') for x in rows),rank
 assert all(x['world_size']==8 and x['shape'][0]==batch and math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) for x in rows)
 assert all(not x.get('optimizer_step_skipped',False) for x in rows)
 required={'aggregator','camera_head','dense_head','language_projector','bridge','action_expert'}
 assert required <= rows[-1]['gradient_elements'].keys(),rank
 assert all(rows[-1]['gradient_elements'][k]>0 for k in required),rank
 # At least one sampled trainable parameter must change across updates.
 assert any(rows[0]['probes'][k]['values']!=rows[-1]['probes'][k]['values'] for k in rows[0]['probes']),rank
 reports.append({'rank':rank,'steps':4,'microbatches_completed':rows[-1]['microbatches_completed'],'peak_memory_bytes':max(x['peak_memory'] for x in rows),'final_loss':rows[-1]['loss'],'elapsed_seconds':rows[-1]['elapsed_seconds'],'gradient_elements':rows[-1]['gradient_elements']})
ckpt=p/'checkpoints/weights/step_000004.pt';assert ckpt.is_file(),ckpt
log=(p/'train.log').read_text()
assert 'Traceback (most recent call last)' not in log and 'Exception ignored in atexit' not in log, 'Training/cleanup exception found'
assert 'Reconstruction gradient audit' in log or 'reconstruction gradient audit' in log
schedule=int(re.search(r'max_steps=(\d+)',log).group(1))
assert schedule>4,'Smoke incorrectly shortened the formal schedule'
(p/'smoke_report.json').write_text(json.dumps({'passed':True,'model_name':'e05','model_factory':'e05.create_e05','formal_schedule_steps':schedule,'stop_after_steps':4,'batch_per_gpu':batch,'gradient_accumulation_steps':accum,'effective_batch':8*batch*accum,'reduced_batch':batch<8,'increased_batch':batch>8,'original_batch8_validated':batch==8 and accum==1,'initialization':'public_vggt_random_actiondit_bridge','ranks':reports,'checkpoint':str(ckpt)},indent=2))
print('PASS: 8 ranks / 4 optimizer steps / checkpoint saved')
