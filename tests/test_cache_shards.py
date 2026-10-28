import io,zipfile
import torch
import pytest
from fastwam.datasets.cache_io import load_cache,cache_exists
from fastwam.datasets.lerobot.vggt_reconstruction_cache import VGGTReconstructionTargetCache

def save_bytes(payload):
 b=io.BytesIO();torch.save(payload,b);return b.getvalue()

@pytest.mark.parametrize('name,shard',[('00000999.pt','000000'),('00001000.pt','000001'),('ab123.t5_len128.wan22ti2v5b.pt','ab')])
def test_archive_and_directory_are_equal(tmp_path,name,shard):
 payload={'context':torch.rand(128,8),'mask':torch.arange(128)<10}
 data=save_bytes(payload);(tmp_path/'shards').mkdir()
 with zipfile.ZipFile(tmp_path/'shards'/f'{shard}.zip','w',zipfile.ZIP_DEFLATED) as z:z.writestr(name,data)
 assert cache_exists(tmp_path/name)
 a=load_cache(tmp_path/name);(tmp_path/name).write_bytes(data);b=load_cache(tmp_path/name)
 for k in payload:assert torch.equal(a[k],b[k])
 assert not cache_exists(tmp_path/'missing.pt')

def test_geometry_reader_accepts_shards_and_rejects_missing(tmp_path):
 payload={'depth':torch.ones(3,4,4),'valid':torch.ones(3,4,4,dtype=torch.bool),'intrinsics':torch.eye(3).repeat(3,1,1),'extrinsics':torch.zeros(3,3,4)}
 (tmp_path/'shards').mkdir()
 with zipfile.ZipFile(tmp_path/'shards/000000.zip','w') as z:z.writestr('00000000.pt',save_bytes(payload))
 c=VGGTReconstructionTargetCache(tmp_path,['high','left','right'],validate=True)
 assert torch.equal(c.get(0)['depth'],payload['depth'])
 with pytest.raises(FileNotFoundError):c.get(1)
