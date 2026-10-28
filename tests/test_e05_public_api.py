from pathlib import Path
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir


def test_public_model_keeps_legacy_checkpoint_identity():
    from e05 import E05
    from fastwam.models.wan22.action_vggt_lang_recon_fastwam import ActionVGGTLangReconFastWAM
    assert E05.__name__ == 'E05'
    assert E05.MODEL_NAME == 'e05'
    assert ActionVGGTLangReconFastWAM is E05
    assert E05.CHECKPOINT_FORMAT == 'action_vggt_lang_recon_fastwam'


def test_public_and_legacy_factories_use_same_implementation(monkeypatch):
    from e05 import create_e05
    from fastwam import runtime
    marker = object()
    calls = []
    def build(**kwargs):
        calls.append(kwargs)
        return marker
    monkeypatch.setattr(runtime, '_create_action_vggt_language_recon_model', build)
    assert create_e05(train_vggt=True) is marker
    assert runtime.create_action_vggt_lang_recon_fastwam(train_vggt=True) is marker
    assert calls == [{'train_vggt': True}, {'train_vggt': True}]


def test_e05_config_is_action_only_and_retains_precision_routing():
    from fastwam.runtime import _VGGT_FP32_BF16_FACTORY_TARGETS
    from experiments.robotwin.fastwam_policy.deploy_policy import _is_action_only_model_config
    cfg = OmegaConf.load('configs/release.yaml')
    assert cfg.model._target_ == 'e05.create_e05'
    assert _is_action_only_model_config(cfg.model)
    assert cfg.model._target_ in _VGGT_FP32_BF16_FACTORY_TARGETS
    with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base='1.3'):
        canonical = compose(config_name='sim_robotwin')
        legacy = compose(config_name='sim_robotwin', overrides=['task=robotwin_langrecon'])
    assert canonical.model._target_ == legacy.model._target_ == 'e05.create_e05'
    assert canonical.EVALUATION.policy_name == 'e05_policy'
    from experiments.robotwin import e05_policy
    assert all(callable(getattr(e05_policy, name)) for name in ('get_model', 'eval', 'reset_model'))
    assert canonical.data.train.processor.vggt_camera_keys == legacy.data.train.processor.vggt_camera_keys
