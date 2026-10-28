import logging
import hashlib
import json
import inspect
import os
import random
import re
from contextlib import contextmanager
from math import ceil, cos, pi
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim
from .models.wan22.vggt_mot import validate_vggt_images

logger = get_logger(__name__)


@contextmanager
def _preserve_rng_state():
    """Keep evaluation from perturbing any RNG stream used by training."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.vggt_learning_rate = float(cfg.get("vggt_learning_rate", 1.0e-6))
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.persistent_workers = (
            bool(cfg.get("persistent_workers", True)) if self.num_workers > 0 else False
        )
        self.prefetch_factor = int(cfg.get("prefetch_factor", 2))
        configured_pin_memory = cfg.get("pin_memory", None)
        self.pin_memory = (
            torch.cuda.is_available()
            if configured_pin_memory is None
            else bool(configured_pin_memory)
        )
        configured_pin_memory_device = cfg.get("pin_memory_device", None)
        self.pin_memory_device = (
            None
            if configured_pin_memory_device in (None, "")
            else str(configured_pin_memory_device)
        )
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.stop_after_steps = cfg.get("stop_after_steps")
        if self.stop_after_steps is not None:
            self.stop_after_steps = int(self.stop_after_steps)
            if self.stop_after_steps <= 0:
                raise ValueError("stop_after_steps must be a positive absolute step.")
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        configured_save_weights_every = cfg.get("save_weights_every", None)
        self.save_weights_every = (
            self.save_every
            if configured_save_weights_every is None
            else int(configured_save_weights_every)
        )
        configured_save_state_every = cfg.get("save_state_every", None)
        self.save_state_every = (
            self.save_every
            if configured_save_state_every is None
            else int(configured_save_state_every)
        )
        self.eval_every = int(cfg.eval_every)
        configured_eval_rollout_every = cfg.get("eval_rollout_every", None)
        if configured_eval_rollout_every is None:
            self.eval_rollout_every = 5 * self.eval_every if self.eval_every > 0 else 0
        else:
            self.eval_rollout_every = int(configured_eval_rollout_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.verify_reconstruction_gradients = bool(
            cfg.get("verify_reconstruction_gradients", False)
        )
        self._reconstruction_gradients_verified = False
        self._reconstruction_gradient_audit = None
        self.seed = int(cfg.seed)

        if self.num_workers < 0:
            raise ValueError("`num_workers` must be non-negative.")
        if self.prefetch_factor < 1:
            raise ValueError("`prefetch_factor` must be at least 1.")
        for interval_name in (
            "log_every",
            "save_every",
            "save_weights_every",
            "save_state_every",
            "eval_every",
            "eval_rollout_every",
        ):
            if getattr(self, interval_name) < 0:
                raise ValueError(f"`{interval_name}` must be non-negative.")
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        self.optimizer_grouping = str(cfg.get("optimizer_grouping", "dtype"))
        optimizer_parameters = self._optimizer_parameters(
            self.model,
            learning_rate=self.learning_rate,
            vggt_learning_rate=self.vggt_learning_rate,
            grouping=self.optimizer_grouping,
        )
        self.optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        self.optimizer_layout = self._optimizer_layout(
            self.model, self.optimizer.param_groups, self.optimizer_grouping
        )
        expected_dtypes = [(p, p.dtype) for group in self.optimizer.param_groups
                           for p in group["params"]]
        logger.info("Optimizer layout: %s", self.optimizer_layout)
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.reconstruction_optimizer_steps = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self._validate_parameter_dtypes(expected_dtypes, self.optimizer_grouping)
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()
        self._validate_parameter_dtypes(expected_dtypes, self.optimizer_grouping)

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        # Explicitly reuse the original run when resuming so W&B does not create
        # a second run with a duplicated name.
        wandb_id = self.cfg.wandb.get("id", None)
        wandb_resume = self.cfg.wandb.get("resume", None)
        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            id=None if wandb_id in (None, "", "null") else str(wandb_id),
            resume=None if wandb_resume in (None, "", "null") else str(wandb_resume),
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
            official_augmentation=getattr(dataset, "vggt_official_augmentation", None) is not None,
        )
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": False,
            "sampler": self.train_sampler,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "worker_init_fn": worker_init_fn,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            if self.pin_memory_device is not None:
                loader_kwargs["pin_memory_device"] = self.pin_memory_device
        return DataLoader(
            **loader_kwargs,
        )

    def _write_smoke_audit(self, sample, loss, metrics, grad_norm):
        """Opt-in real-run evidence; does not alter sampling or model RNGs."""
        audit_dir = os.environ.get("FASTWAM_SMOKE_AUDIT_DIR")
        if not audit_dir:
            return
        model = self.accelerator.unwrap_model(self.model)
        if not torch.isfinite(loss).all():
            raise RuntimeError("Smoke audit: non-finite loss.")
        probes = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and any(part in name for part in ("action_expert", "camera_head", "dense_head", "bridge", "aggregator", "language_projector")):
                group = next(part for part in ("action_expert", "camera_head", "dense_head", "bridge", "aggregator", "language_projector") if part in name)
                if group not in probes:
                    probes[group] = {"name": name, "values": parameter.detach().flatten()[:32].float().cpu().tolist()}
        rgb = sample["vggt_images"].detach().float().cpu().contiguous()
        record = {"rank": self.accelerator.process_index, "world_size": self.accelerator.num_processes,
                  "pytorch_allocator": os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")),
                  "model_name": getattr(model, "MODEL_NAME", None), "model_class": type(model).__module__ + "." + type(model).__name__,
                  "device": str(self.accelerator.device), "step": self.global_step,
                  "gradient_accumulation_steps": self.gradient_accumulation_steps, "microbatches_completed": self.batch_in_epoch,
                  "shape": list(rgb.shape), "rgb_sha256": hashlib.sha256(rgb.numpy().tobytes()).hexdigest(),
                  "loss": float(loss.detach()), "metrics": {key: float(value.detach()) if isinstance(value, torch.Tensor) else float(value) for key, value in metrics.items()},
                  "grad_norm": float(grad_norm) if grad_norm is not None else None,
                  "peak_memory": torch.cuda.max_memory_allocated(), "elapsed_seconds": time.perf_counter() - self.run_start_time, "gradient_elements": getattr(self, "_smoke_gradient_counts", {}), "probes": probes, "optimizer_step_skipped": self.accelerator.optimizer_step_was_skipped}
        if "vggt_augmentation_trace" in sample:
            record["augmentation"] = sample["vggt_augmentation_trace"].detach().cpu().tolist()
        path = Path(audit_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"rank_{self.accelerator.process_index}.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            if bool(getattr(self.model, "train_vggt", False)):
                def cosine_multiplier(step: int) -> float:
                    progress = min(max(float(step) / remaining_steps, 0.0), 1.0)
                    return 0.01 + 0.99 * 0.5 * (1.0 + cos(pi * progress))

                main_scheduler = LambdaLR(self.optimizer, lr_lambda=cosine_multiplier)
            else:
                main_scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=remaining_steps,
                    eta_min=self.learning_rate * 0.01,
                )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        # ZeRO builds detached FP32 master partitions during prepare(). Loading
        # only model weights afterwards must refresh those partitions, otherwise
        # the first optimizer step restores the pre-checkpoint initialization.
        refresh = getattr(getattr(self.model, "optimizer", None), "refresh_fp32_params", None)
        if refresh is not None:
            refresh()
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Applying configured training modes; train_vggt=%s.", getattr(self.accelerator.unwrap_model(self.model), "train_vggt", False))
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
        if bool(getattr(model, "train_vggt", False)):
            backbone = getattr(model, "vggt_backbone", None)
            if backbone is None or model._modules.get("vggt_backbone") is not backbone:
                raise RuntimeError(
                    "Trainable VGGT must be registered as `model.vggt_backbone`."
                )
            if not bool(getattr(backbone, "trainable", False)):
                raise RuntimeError(
                    "`model.vggt_backbone` was not constructed in trainable mode."
                )
            backbone.train()
            backbone.requires_grad_(True)

    @staticmethod
    def _optimizer_parameters(model, *, learning_rate: float, vggt_learning_rate: float,
                              grouping: str = "dtype"):
        if grouping not in {"dtype", "legacy"}:
            raise ValueError("optimizer_grouping must be 'dtype' or 'legacy'.")
        groups = Wan22Trainer._logical_optimizer_parameters(
            model, learning_rate=learning_rate, vggt_learning_rate=vggt_learning_rate)
        if grouping == "legacy":
            return groups
        if not groups or not isinstance(groups[0], dict):
            groups = [{"params": groups, "lr": float(learning_rate)}]
        result, seen = [], set()
        for logical_name, group in zip(("main", "vggt"), groups):
            buckets = {}
            for parameter in group["params"]:
                if id(parameter) in seen:
                    raise RuntimeError("Optimizer parameter groups contain duplicate parameters.")
                seen.add(id(parameter))
                buckets.setdefault(parameter.dtype, []).append(parameter)
            for dtype, parameters in buckets.items():
                result.append({**group, "params": parameters,
                               "logical_group": logical_name, "parameter_dtype": str(dtype)})
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        if seen != expected:
            raise RuntimeError("Optimizer groups do not cover exactly the trainable model parameters.")
        return result

    @staticmethod
    def _optimizer_layout(model, groups, policy):
        names = {id(p): name for name, p in model.named_parameters()}
        layout = []
        for index, group in enumerate(groups):
            entries = [(names[id(p)], list(p.shape), str(p.dtype)) for p in group["params"]]
            layout.append({
                "logical_group": group.get("logical_group", "main" if index == 0 else "vggt"),
                "signature": hashlib.sha256(json.dumps(entries).encode()).hexdigest(),
                "dtypes": sorted({entry[2] for entry in entries}),
                "numel": sum(p.numel() for p in group["params"]),
            })
        return {"version": 1, "policy": policy, "groups": layout}

    @staticmethod
    def _validate_parameter_dtypes(expected, policy):
        changed = [(str(dtype), str(p.dtype), p.numel()) for p, dtype in expected if p.dtype != dtype]
        if changed and policy == "dtype":
            raise RuntimeError(f"Model parameter dtype changed during prepare/resume: {changed[:8]}")
        if changed:
            logger.warning("Legacy grouping changed %d parameter tensors' dtype.", len(changed))

    def _validate_optimizer_layout(self, metadata):
        current = getattr(self, "optimizer_layout", None)
        if current is None:  # Lightweight callers that do not construct an optimizer.
            return
        saved = metadata.get("optimizer_layout")
        if saved is None:
            if current["policy"] != "legacy":
                raise ValueError("Old full-state checkpoint requires optimizer_grouping=legacy; "
                                 "use a .pt weights checkpoint to initialize dtype grouping.")
        elif saved != current:
            raise ValueError("Full-state resume requires identical optimizer grouping/layout; "
                             "use a .pt checkpoint for weights-only initialization.")

    def _optimizer_learning_rates(self):
        rates = {}
        for index, group in enumerate(self.optimizer.param_groups):
            name = group.get("logical_group", "main" if index == 0 else "vggt")
            rate = float(group["lr"])
            if name in rates and rates[name] != rate:
                raise RuntimeError(f"Learning rates diverged within logical optimizer group {name}.")
            rates[name] = rate
        return rates.get("main", float(self.optimizer.param_groups[0]["lr"])), rates.get("vggt")

    @staticmethod
    def _logical_optimizer_parameters(model, *, learning_rate: float, vggt_learning_rate: float):
        trainable_params = list(model.dit.parameters())
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        trainable_params = [
            parameter for parameter in trainable_params if parameter.requires_grad
        ]
        if not bool(getattr(model, "train_vggt", False)):
            return trainable_params

        backbone = getattr(model, "vggt_backbone", None)
        if backbone is None or model._modules.get("vggt_backbone") is not backbone:
            raise RuntimeError(
                "Trainable VGGT must be registered as `model.vggt_backbone` "
                "before optimizer initialization."
            )
        vggt_params = [
            parameter for parameter in backbone.parameters() if parameter.requires_grad
        ]
        if not vggt_params:
            raise RuntimeError("Trainable VGGT exposes no trainable parameters.")
        main_ids = {id(parameter) for parameter in trainable_params}
        if main_ids.intersection(id(parameter) for parameter in vggt_params):
            raise RuntimeError("VGGT and main optimizer parameter groups overlap.")
        return [
            {"params": trainable_params, "lr": float(learning_rate)},
            {"params": vggt_params, "lr": float(vggt_learning_rate)},
        ]

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        vggt_images = sample.get("vggt_images", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if vggt_images is not None:
            if not isinstance(vggt_images, torch.Tensor):
                raise TypeError(f"`sample['vggt_images']` must be a torch.Tensor, got {type(vggt_images)}")
            if vggt_images.ndim == 4:
                vggt_images = vggt_images.unsqueeze(0)
            validate_vggt_images(vggt_images, batch_size=int(video.shape[0]))

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            **({"vggt_images": vggt_images} if vggt_images is not None else {}),
            "action_horizon": action_horizon,
        }

    @staticmethod
    def _to_batched_action_eval_sample(
        sample,
        *,
        require_future_vggt: bool = False,
        require_teacher_features: bool = False,
        require_depth_targets: bool = False,
        require_geometry_latent_targets: bool = False,
    ):
        required = {"action", "context", "context_mask", "proprio", "vggt_images"}
        if require_future_vggt:
            required.add("vggt_future_images")
        if require_teacher_features:
            required.update({"vggt_teacher_g0", "vggt_teacher_g32"})
        if require_depth_targets:
            required.update({"depth_target", "depth_weight", "depth_future_is_pad"})
        if require_geometry_latent_targets:
            required.add("vggt_latent_future")
        missing = sorted(required - set(sample))
        if missing:
            raise ValueError(f"Action-only eval sample is missing {missing}.")

        result = {}
        expected_dims = {
            "action": 2,
            "context": 2,
            "context_mask": 1,
            "proprio": 2,
            "vggt_images": 4,
            "language_context_mask": 1,
            # A future VGGT clip is [T,V,3,H,W] for an unbatched sample
            # (currently T=9), while a single future frame is [V,3,H,W].
            # Handle this key specially below so the temporal dimension is
            # not mistaken for the batch dimension.
            "vggt_future_images": 4,
            "vggt_teacher_g0": 2,
            "vggt_teacher_g32": 2,
            "vggt_future_is_pad": 0,
            "action_is_pad": 1,
            "depth_target": 4,
            "depth_weight": 4,
            "depth_future_is_pad": 1,
            "vggt_latent_future": 4,
        }
        for key, unbatched_dim in expected_dims.items():
            value = sample.get(key)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Action-only eval `{key}` must be a tensor, got {type(value)}."
                )
            if key == "vggt_future_images" and value.ndim == 5:
                # Unbatched future clip: [T,V,3,H,W] -> [1,T,V,3,H,W].
                value = value.unsqueeze(0)
            elif value.ndim == unbatched_dim:
                value = value.unsqueeze(0)
            expected_ndim = 6 if key == "vggt_future_images" and value.ndim == 6 else unbatched_dim + 1
            if value.ndim != expected_ndim:
                accepted_dims = "4, 5, or 6" if key == "vggt_future_images" else f"{unbatched_dim} or {unbatched_dim + 1}"
                raise ValueError(
                    f"Action-only eval `{key}` must have {accepted_dims} dims, "
                    f"got {tuple(value.shape)}."
                )
            result[key] = value
        validate_vggt_images(
            result["vggt_images"],
            batch_size=int(result["action"].shape[0]),
        )
        if "vggt_future_images" in result:
            future_images = result["vggt_future_images"]
            if future_images.ndim == 6:
                # Validate one frame per batch; the model validates the same
                # view layout before encoding the complete clip.
                validate_vggt_images(
                    future_images[:, 0],
                    batch_size=int(result["action"].shape[0]),
                )
            else:
                validate_vggt_images(
                    future_images,
                    batch_size=int(result["action"].shape[0]),
                )
        return result

    def _denormalize_action_for_eval(
        self, raw_action: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        if raw_action.ndim == 2:
            action_btd = raw_action.unsqueeze(0)
        elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
            action_btd = raw_action
        else:
            raise ValueError(
                "Action eval tensor must have shape [T,D] or [1,T,D], got "
                f"{tuple(raw_action.shape)}."
            )
        action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)
        proprio = proprio.detach().to(device="cpu", dtype=torch.float32)
        processor = self.val_dataset.lerobot_dataset.processor
        batch = {"action": action_btd, "state": proprio}
        batch = processor.action_state_merger.backward(batch)
        batch = processor.normalizer.backward(batch)
        action_meta = processor.shape_meta["action"]
        state_meta = processor.shape_meta["state"]
        merged_batch = {
            "action": {
                meta["key"]: batch["action"][meta["key"]].squeeze(0)
                for meta in action_meta
            },
            "state": {
                meta["key"]: batch["state"][meta["key"]].squeeze(0)
                for meta in state_meta
            },
        }
        denormalized = processor.action_state_merger.forward(merged_batch)[
            "action"
        ].unsqueeze(0)
        if denormalized.ndim != 3 or denormalized.shape[0] != 1:
            raise ValueError(
                "Denormalized action must have shape [1,T,D], got "
                f"{tuple(denormalized.shape)}."
            )
        return denormalized

    @torch.no_grad()
    def _evaluate_action_only(self, *, run_rollout: bool):
        model = self.accelerator.unwrap_model(self.model)
        rng = torch.Generator(device="cpu").manual_seed(
            self.global_step + self.accelerator.process_index
        )
        eval_index = torch.randint(
            0, len(self.val_dataset), (1,), generator=rng
        ).item()
        sample = self._to_batched_action_eval_sample(
            self.val_dataset[eval_index],
            require_future_vggt=bool(
                getattr(model, "requires_vggt_future_images", False)
            ),
            require_teacher_features=bool(
                getattr(model, "requires_vggt_teacher_features", False)
            ),
            require_depth_targets=bool(
                getattr(model, "requires_depth_targets", False)
            ),
            require_geometry_latent_targets=bool(
                getattr(model, "requires_vggt_geometry_latent_targets", False)
            ),
        )

        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
        gathered_loss = self.accelerator.gather_for_metrics(
            val_loss.detach().float().reshape(1)
        ).mean()
        result = {"val_loss": float(gathered_loss.cpu().item())}
        if not run_rollout:
            return result

        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return result

        infer_kwargs = {
            "action_horizon": int(sample["action"].shape[1]),
            "vggt_images": sample["vggt_images"],
            "context": sample["context"],
            "context_mask": sample["context_mask"],
            "proprio": sample["proprio"][:, 0],
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
        }
        if "vggt_teacher_g0" in sample:
            infer_kwargs["vggt_teacher_g0"] = sample["vggt_teacher_g0"]
        pred_action = model.infer_action(**infer_kwargs)
        pred_denorm = self._denormalize_action_for_eval(
            pred_action, sample["proprio"]
        )
        gt_denorm = self._denormalize_action_for_eval(
            sample["action"], sample["proprio"]
        )
        if pred_denorm.shape != gt_denorm.shape:
            raise ValueError(
                "Predicted/GT action shape mismatch after denormalization: "
                f"{tuple(pred_denorm.shape)} vs {tuple(gt_denorm.shape)}."
            )
        diff = pred_denorm - gt_denorm
        result.update(
            {
                "action_l1": float(diff.abs().mean().item()),
                "action_l2": float(diff.pow(2).mean().item()),
            }
        )
        self.accelerator.wait_for_everyone()
        return result

    @torch.no_grad()
    def evaluate(self, *, run_rollout: bool = True):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        processor = None
        if getattr(self.val_dataset, "vggt_official_augmentation", None) is not None:
            processor = self.val_dataset.lerobot_dataset.processor
        was_processor_training = processor.is_train if processor is not None else False
        with _preserve_rng_state():
            model.eval()
            try:
                if processor is not None:
                    processor.eval()
                if getattr(model, "evaluation_mode", None) == "action_only":
                    return self._evaluate_action_only(run_rollout=run_rollout)
                return self._evaluate_video(run_rollout=run_rollout)
            finally:
                if processor is not None and was_processor_training:
                    processor.train()
                if was_dit_training:
                    self._set_dit_only_train_mode()

    @torch.no_grad()
    def _evaluate_video(self, *, run_rollout: bool):
        model = self.accelerator.unwrap_model(self.model)

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
        gathered_loss = self.accelerator.gather_for_metrics(
            val_loss.detach().float().reshape(1)
        ).mean()
        result = {"val_loss": float(gathered_loss.cpu().item())}
        if not run_rollout:
            return result

        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return result
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if "vggt_images" in sample:
            infer_kwargs["vggt_images"] = sample["vggt_images"]
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        result.update(
            {
                "psnr_rg": float(psnr_rollout_vs_gt),
                "ssim_rg": float(ssim_rollout_vs_gt),
                "psnr_rd": float(psnr_rollout_vs_decode),
                "ssim_rd": float(ssim_rollout_vs_decode),
                "psnr_dg": float(psnr_decode_vs_gt),
                "ssim_dg": float(ssim_decode_vs_gt),
                "video_path": video_path,
            }
        )
        if action_l2 is not None:
            result["action_l2"] = float(action_l2)
        if action_l1 is not None:
            result["action_l1"] = float(action_l1)
        self.accelerator.wait_for_everyone()
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        augmentation = getattr(getattr(self, "train_dataset", None), "vggt_official_augmentation", None)
        if augmentation is not None:
            payload["vggt_official_augmentation"] = {
                **augmentation.contract(), "batch_size": self.batch_size,
                "world_size": self.accelerator.num_processes, "training_seed": self.seed,
            }
        if hasattr(self, "optimizer_layout"):
            payload["optimizer_layout"] = self.optimizer_layout
        model = self.accelerator.unwrap_model(self.model)
        contract = getattr(model, "reconstruction_loss_contract", lambda: None)()
        if contract is not None:
            payload["reconstruction_loss"] = contract
            payload["reconstruction_optimizer_steps"] = self.reconstruction_optimizer_steps
            payload["reconstruction_schedule_steps"] = self.max_steps
        if hasattr(model, "sharing_config"):
            payload["model_sharing"] = model.sharing_config()
            payload["world_size"] = self.accelerator.num_processes
        if hasattr(model, "high_noise_reconstruction_target"):
            payload["high_noise_reconstruction_target"] = model.high_noise_reconstruction_target
            payload["high_noise_use_language_condition"] = model.vggt_backbone.high_noise_use_language_condition
        if hasattr(model, "low_noise_reconstruction_target"):
            payload["low_noise_reconstruction_target"] = model.low_noise_reconstruction_target
        if hasattr(model, "validate_training_state_metadata"):
            payload["model_architecture"] = model.checkpoint_metadata()
            payload["world_size"] = self.accelerator.num_processes
        if hasattr(model, "route_action_weighting_config"):
            payload["route_action_weighting"] = model.route_action_weighting_config()
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self, *, save_weights: bool = True, save_state: bool = True):
        step_tag = f"step_{self.global_step:06d}"

        ckpt_path = None
        if save_weights:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
            self.accelerator.wait_for_everyone()

        state_path = None
        if save_state:
            state_path = os.path.join(self.state_dir, step_tag)
            ensure_dir(state_path)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def _reduce_training_metrics(self, loss, loss_dict, grad_norm):
        metric_keys = sorted(loss_dict)
        metric_tensors = [loss.detach().to(dtype=torch.float32).reshape(())]
        for key in metric_keys:
            value = loss_dict[key]
            value_tensor = torch.as_tensor(
                value,
                device=loss.device,
                dtype=torch.float32,
            )
            if value_tensor.numel() != 1:
                raise ValueError(
                    f"Training metric `{key}` must be scalar, got shape "
                    f"{tuple(value_tensor.shape)}."
                )
            metric_tensors.append(value_tensor.detach().reshape(()))
        grad_norm_tensor = torch.as_tensor(
            grad_norm,
            device=loss.device,
            dtype=torch.float32,
        )
        if grad_norm_tensor.numel() != 1:
            raise ValueError(
                "Gradient norm must be scalar, got shape "
                f"{tuple(grad_norm_tensor.shape)}."
            )
        metric_tensors.append(grad_norm_tensor.detach().reshape(()))
        reduced = self.accelerator.reduce(
            torch.stack(metric_tensors),
            reduction="mean",
        )
        values = reduced.cpu().tolist()
        return (
            float(values[0]),
            {key: float(value) for key, value in zip(metric_keys, values[1:-1])},
            float(values[-1]),
        )

    def load_training_state(self, state_dir: str):
        model = self.accelerator.unwrap_model(self.model)
        contract = getattr(model, "reconstruction_loss_contract", lambda: None)()
        metadata_path = Path(state_dir) / "trainer_state.json"
        saved_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        augmentation = getattr(getattr(self, "train_dataset", None), "vggt_official_augmentation", None)
        expected_augmentation = None if augmentation is None else {
            **augmentation.contract(), "batch_size": self.batch_size,
            "world_size": self.accelerator.num_processes, "training_seed": self.seed,
        }
        if saved_metadata.get("vggt_official_augmentation") != expected_augmentation:
            raise ValueError("Full-state resume requires unchanged official augmentation; use weights-only initialization instead.")
        if saved_metadata.get("reconstruction_loss") != contract:
            raise ValueError("Full-state resume requires the same reconstruction loss backend/config.")
        if hasattr(model, "validate_training_state_metadata"):
            model.validate_training_state_metadata(saved_metadata, self.accelerator.num_processes)
        if contract is not None:
            if saved_metadata.get("reconstruction_schedule_steps") != self.max_steps:
                raise ValueError("Official loss resume requires unchanged max_steps for intrinsics warmup.")
            self.reconstruction_optimizer_steps = saved_metadata["reconstruction_optimizer_steps"]
            model.set_reconstruction_loss_progress(self.reconstruction_optimizer_steps / self.max_steps)
        if hasattr(model, "sharing_config"):
            metadata_path = Path(state_dir) / "trainer_state.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, encoding="utf-8") as stream:
                    metadata = json.load(stream)
            if hasattr(model, "high_noise_reconstruction_target"):
                source_target = metadata.get("high_noise_reconstruction_target", "current")
                if source_target != model.high_noise_reconstruction_target:
                    raise ValueError("Full-state resume requires the same high-noise reconstruction target.")
                source_language = metadata.get("high_noise_use_language_condition", False)
                if source_language != model.vggt_backbone.high_noise_use_language_condition:
                    raise ValueError("Full-state resume requires the same high-noise language setting.")
            if hasattr(model, "low_noise_reconstruction_target"):
                if metadata.get("low_noise_reconstruction_target", "future") != model.low_noise_reconstruction_target:
                    raise ValueError("Full-state resume requires the same low-noise reconstruction target.")
            target = model.sharing_config()
            source = metadata.get("model_sharing", {key: True for key in target})
            if source != target:
                raise ValueError("Full-state resume requires identical sharing; use a .pt checkpoint with expand_shared for weights-only initialization.")
            if metadata.get("world_size", self.accelerator.num_processes) != self.accelerator.num_processes:
                raise ValueError("Full-state resume requires the same distributed world size.")
        self._validate_optimizer_layout(saved_metadata)
        if hasattr(model, "route_action_weighting_config"):
            if saved_metadata.get("route_action_weighting") != model.route_action_weighting_config():
                raise ValueError("Full-state resume requires the same route action weighting configuration.")
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def _install_reconstruction_gradient_audit(self, model):
        """Observe gradients before ZeRO moves them out of ``parameter.grad``."""
        backbone = getattr(model, "vggt_backbone", None)
        modules = {
            "aggregator": getattr(backbone, "aggregator", None),
            "camera_head": getattr(backbone, "camera_head", None),
            "dense_head": getattr(backbone, "dense_head", None),
        }
        if getattr(model, "CHECKPOINT_FORMAT", None) in {
            "action_vggt_loop_recon_fastwam", "action_vggt_lang_recon_fastwam",
        }:
            modules.update(language_projector=backbone.language_projector,
                           bridge=model.mot.vggt_bridge, action_expert=model.action_expert)
        if getattr(model, "CHECKPOINT_FORMAT", None) == "action_vggt_moe_recon_fastwam":
            modules.update(
                low_frame_blocks=backbone.low_noise_vggt.frame_blocks,
                low_inter_frame_blocks=backbone.low_noise_vggt.inter_frame_blocks,
                language_projector=backbone.language_projector,
            )
        if getattr(model, "CHECKPOINT_FORMAT", None) == "action_vggt_mot_recon_fastwam":
            modules = {}
            for name in ("va1", "va2"):
                stream = getattr(backbone, name)
                modules.update({f"{name}_patch": stream.patch_embed,
                                f"{name}_frame": stream.frame_blocks,
                                f"{name}_inter_frame": stream.inter_frame_blocks,
                                f"{name}_tokens": torch.nn.ParameterList([stream.camera_token, stream.register_token])})
            for route in ("high", "low"):
                camera, dense = backbone.geometry_heads(route)
                modules.update({f"{route}_camera_head": camera, f"{route}_dense_head": dense})
            if backbone.va1_use_language_condition or backbone.va2_use_language_condition:
                modules["language_projector"] = backbone.language_projector
            bridge = model.mot.vggt_bridge
            if model.share_bridge:
                modules["bridge"] = bridge
            else:
                for source in ("va1", "va2"):
                    if backbone.action_vggt_source in ("both", source):
                        modules[f"{source}_bridge"] = getattr(bridge, source)
        if getattr(model, "CHECKPOINT_FORMAT", None) == "action_vggt_multistage_recon_fastwam":
            modules = {"action_expert": model.action_expert, "bridge": model.mot.vggt_bridge}
            for name, expert in backbone.experts.items():
                modules.update({f"{name}_frame": expert.frame_blocks,
                                f"{name}_inter_frame": expert.inter_frame_blocks,
                                f"{name}_patch": expert.patch_embed})
                camera, dense = backbone.geometry_heads(name)
                modules.update({f"{name}_camera_head": camera, f"{name}_dense_head": dense})
            if backbone.language_projector is not None:
                modules["language_projector"] = backbone.language_projector
        audit = {}
        for name, module in modules.items():
            if module is None:
                raise RuntimeError(
                    f"Reconstruction gradient audit cannot find `{name}`."
                )
            state = {"tensors": 0, "elements": 0, "finite": None, "handles": []}

            def observe(gradient, *, state=state):
                state["tensors"] += 1
                state["elements"] += gradient.numel()
                is_finite = torch.isfinite(gradient).all()
                if state["finite"] is None:
                    state["finite"] = is_finite
                else:
                    state["finite"].logical_and_(is_finite)
                return gradient

            for parameter in module.parameters():
                if parameter.requires_grad:
                    state["handles"].append(parameter.register_hook(observe))
            if not state["handles"]:
                raise RuntimeError(
                    f"Reconstruction gradient audit found no trainable parameters for `{name}`."
                )
            audit[name] = state
        self._reconstruction_gradient_audit = audit

    def _verify_reconstruction_gradients(self):
        audit = self._reconstruction_gradient_audit
        if audit is None:
            raise RuntimeError("Reconstruction gradient audit was not installed.")
        counts = {}
        for name, state in audit.items():
            if state["tensors"] == 0 or state["finite"] is None:
                raise RuntimeError(
                    f"Reconstruction gradient audit observed no gradients for `{name}`."
                )
            if not bool(state["finite"].item()):
                raise FloatingPointError(
                    f"Reconstruction gradient audit found NaN/Inf in `{name}`."
                )
            counts[name] = state["elements"]
            for handle in state["handles"]:
                handle.remove()
        self._reconstruction_gradient_audit = None
        return counts

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if getattr(self, "verify_reconstruction_gradients", False):
            self._install_reconstruction_gradient_audit(unwrapped_model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        stop_at = getattr(self, "stop_after_steps", None)
        if stop_at is not None and stop_at <= self.global_step:
            raise ValueError("stop_after_steps must be greater than the resumed global step.")
        self.train_sampler.set_epoch(self.epoch)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                self.train_sampler.set_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                # All accumulated microbatches see the same progress. Count only
                # successful optimizer updates, matching the LR scheduler below.
                if getattr(unwrapped_model, "reconstruction_loss_contract", lambda: None)() is not None:
                    unwrapped_model.set_reconstruction_loss_progress(
                        self.reconstruction_optimizer_steps / self.max_steps)
                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    if (
                        getattr(self, "verify_reconstruction_gradients", False)
                        and not getattr(self, "_reconstruction_gradients_verified", False)
                    ):
                        gradient_counts = self._verify_reconstruction_gradients()
                        self._reconstruction_gradients_verified = True
                        self._smoke_gradient_counts = gradient_counts
                        if self.accelerator.is_main_process:
                            logger.info(
                                "Reconstruction gradient audit passed: %s",
                                " ".join(
                                    f"{name}={count}" for name, count in gradient_counts.items()
                                ),
                            )
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                        if getattr(unwrapped_model, "reconstruction_loss_contract", lambda: None)() is not None:
                            self.reconstruction_optimizer_steps += 1
                            unwrapped_model.set_reconstruction_loss_progress(
                                self.reconstruction_optimizer_steps / self.max_steps)
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    self._write_smoke_audit(sample, loss, loss_dict, grad_norm)
                    if self.log_every > 0 and self.global_step % self.log_every == 0:
                        global_loss, global_loss_metrics, global_grad_norm = (
                            self._reduce_training_metrics(loss, loss_dict, grad_norm)
                        )
                        if self.accelerator.is_main_process:
                            current_lr, current_vggt_lr = self._optimizer_learning_rates()
                            eta_str, steps_per_sec = self._estimate_eta()
                            description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                                self.epoch,
                                self.global_step,
                                self.max_steps,
                                global_loss,
                            )
                            if global_loss_metrics:
                                detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                                description += detail_str + " "
                            if current_vggt_lr is not None:
                                description += "vggt_lr=%.2e " % current_vggt_lr
                            description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                                current_lr,
                                steps_per_sec,
                                steps_per_sec * self.batch_size * self.accelerator.num_processes,
                                eta_str,
                            )
                            logger.info(description)

                            wandb_payload = {
                                "train/loss": global_loss,
                                "train/grad_norm": global_grad_norm,
                                "train/lr": current_lr,
                                "performance/steps_per_sec": steps_per_sec,
                                "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            }
                            if current_vggt_lr is not None:
                                wandb_payload["train/vggt_lr"] = current_vggt_lr
                            for key, value in global_loss_metrics.items():
                                wandb_payload[f"train/{key}"] = value
                            self._wandb_log(wandb_payload)

                    eval_loss_due = (
                        self.eval_every > 0
                        and self.global_step % self.eval_every == 0
                    )
                    eval_rollout_due = (
                        self.eval_rollout_every > 0
                        and self.global_step % self.eval_rollout_every == 0
                    )
                    if self.val_dataset is not None and (
                        eval_loss_due or eval_rollout_due
                    ):
                        metrics = self.evaluate(run_rollout=eval_rollout_due)
                        if metrics is not None and self.accelerator.is_main_process:
                            if "psnr_rd" not in metrics:
                                description = "[eval] step=%d val_loss=%.4f" % (
                                    self.global_step,
                                    metrics["val_loss"],
                                )
                            else:
                                description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                    self.global_step,
                                    metrics["val_loss"],
                                    metrics["psnr_rd"],
                                    metrics["ssim_rd"],
                                )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                            }
                            for key in (
                                "psnr_rg",
                                "ssim_rg",
                                "psnr_rd",
                                "ssim_rd",
                                "psnr_dg",
                                "ssim_dg",
                            ):
                                if key in metrics:
                                    eval_payload[f"eval/{key}"] = float(metrics[key])
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    is_final_step = self.global_step >= self.max_steps
                    is_stop_step = stop_at is not None and self.global_step >= stop_at
                    save_weights_due = (
                        self.save_weights_every > 0
                        and self.global_step % self.save_weights_every == 0
                    )
                    save_state_due = (
                        self.save_state_every > 0
                        and self.global_step % self.save_state_every == 0
                    )
                    if is_final_step or is_stop_step:
                        save_weights_due = True
                        save_state_due = (True if is_final_step else bool(self.cfg.get("save_state_at_stop", True)))
                    if save_weights_due or save_state_due:
                        ckpt_info = self.save_checkpoint(
                            save_weights=save_weights_due,
                            save_state=save_state_due,
                        )
                        if self.accelerator.is_main_process:
                            logger.info(
                                "%s step=%d weights=%s state=%s",
                                "[done] max_steps reached" if is_final_step else ("[stopped] requested step reached" if is_stop_step else "[ckpt]"),
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                    if is_final_step or is_stop_step:
                        # Persistent DataLoader workers must exit while Python
                        # and its IPC services are still alive, before atexit.
                        shutdown = getattr(data_iter, "_shutdown_workers", None)
                        if shutdown is not None:
                            shutdown()
                        self.accelerator.wait_for_everyone()
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
