"""Read unchanged Torch cache files from a directory or independently extractable ZIPs."""
import io,os,zipfile
from pathlib import Path
from functools import lru_cache
import torch

def shard_path(path):
 path=Path(path)
 key=f'{int(path.stem)//1000:06d}' if path.stem.isdecimal() else path.name[:2]
 return path.parent/'shards'/f'{key}.zip'

@lru_cache(maxsize=16)
def _members(path,pid):
 # Cache central-directory metadata, never an open file descriptor across fork.
 with zipfile.ZipFile(path) as z:return {i.filename:i for i in z.infolist()}

def cache_exists(path):
 p=Path(path)
 if p.is_file():return True
 shard=shard_path(p)
 return shard.is_file() and p.name in _members(str(shard),os.getpid())

def load_cache(path):
 p=Path(path)
 shard=shard_path(p)
 if not shard.is_file() and p.is_file():return torch.load(p,map_location='cpu',weights_only=True)
 with zipfile.ZipFile(shard) as z:
  data=z.read(p.name)
 return torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
