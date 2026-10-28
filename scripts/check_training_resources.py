"""Fail before starting the selected 8-GPU recipe when the node is occupied."""
import json,os,sys
from pathlib import Path
import torch
batch=int(os.environ.get('E05_SMOKE_BATCH_SIZE','8'))
assert batch in (16,8,4,2),batch
accum=int(os.environ.get('E05_SMOKE_ACCUM','1'))
assert accum in (1,2,4),accum
minimum=float(os.environ.get('E05_MIN_FREE_GPU_GIB',str({16:88,8:72,4:32,2:24}[batch])))
count=torch.cuda.device_count();free=[torch.cuda.mem_get_info(i)[0]/1024**3 for i in range(min(8,count))]
passed=count>=8 and all(x>=minimum for x in free)
report={'passed':passed,'status':'ready' if passed else 'blocked_resources','required_gpus':8,'batch_per_gpu':batch,'gradient_accumulation_steps':accum,'effective_batch':8*batch*accum,'reduced_batch':batch<8,'increased_batch':batch>8,'visible_gpus':count,'minimum_free_gib_per_gpu':minimum,'free_gib':free,'basis':('Historical batch-8/no-checkpointing run observed approximately 67.2 GiB; 72 GiB is an operational admission threshold, not a measured E05 peak.' if batch==8 else 'User-authorized batch-size attempt. Admission threshold is a heuristic, not a measured memory requirement or guarantee against OOM.'),'training_started':False}
out=Path(os.environ.get('E05_OUTPUT_DIR','runs/resource_check'));out.mkdir(parents=True,exist_ok=True);(out/'resource_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if not passed:sys.exit(75)
