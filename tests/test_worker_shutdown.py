"""Exercise tensor IPC cleanup on the same filesystem as the project temp dir."""
import os
import subprocess
import sys
import textwrap


def test_persistent_worker_shutdown_closes_tensor_ipc(tmp_path):
    program = textwrap.dedent('''
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from fastwam.utils.pytorch_utils import set_global_seed
        loader = DataLoader(
            TensorDataset(torch.arange(160).reshape(40, 4)), batch_size=4,
            num_workers=2, persistent_workers=True,
            worker_init_fn=set_global_seed(42, get_worker_init_fn=True),
        )
        iterator = iter(loader)
        for _ in range(2):
            assert next(iterator)[0].shape == (4, 4)
        iterator._shutdown_workers()
        assert all(not w.is_alive() and w.exitcode == 0 for w in iterator._workers)
        print('workers closed')
    ''')
    env = dict(os.environ, TMPDIR=str(tmp_path), TMP=str(tmp_path), TEMP=str(tmp_path))
    result = subprocess.run([sys.executable, '-c', program], env=env,
                            text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'workers closed' in result.stdout
    assert 'Traceback' not in result.stderr + result.stdout
