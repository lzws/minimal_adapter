#!/usr/bin/env python
"""Minimal latent-backbone prompt adapter trainer."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator

from safe_embedding_adapter.adapter_factory import ADAPTER_TYPES, build_adapter_from_config
from safe_embedding_adapter.conditions import precompute_ip_condition_embeddings
from safe_embedding_adapter.config import DatasetConfig, ProxyLatentConfig, torch_dtype_from_name
from safe_embedding_adapter.data import (
    BalancedPromptBatchSampler,
    PromptSample,
    load_prompt_samples,
    split_prompt_samples,
)
from safe_embedding_adapter.latent_backbone_feedback import (
    DEFAULT_LATENT_BACKBONE_MODEL_FILE,
    LatentBackboneCosineScorer,
    LatentBackboneFeatureEncoder,
    build_reference_prototype_from_prompts,
    load_latent_reference_payload,
    load_latent_reference_prototypes,
)
from safe_embedding_adapter.model import (
    IP_CONDITION_NAMES,
    freeze_module,
    ip_condition_from_metadata,
)
from safe_embedding_adapter.zimage_proxy import ZImageProxyLatentRunner
from safe_embedding_adapter.zimage_train_pipeline import ZImageTrainPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all supported prompt adapter variants.")
    parser.add_argument("--model_path", required=True, help="Local Z-Image-Turbo directory.")
    parser.add_argument("--unsafe_csv", required=True)
    parser.add_argument("--benign_csv", required=True)
    parser.add_argument("--latent_reference_features", required=True)
    parser.add_argument("--latent_backbone_checkpoint", required=True)
    parser.add_argument("--latent_backbone_config", default=None)
    parser.add_argument("--latent_backbone_model_file", default=str(DEFAULT_LATENT_BACKBONE_MODEL_FILE))
    parser.add_argument("--latent_backbone_torch_dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--latent_backbone_input_size", type=int, default=None)
    parser.add_argument("--latent_reference_key", default="Doraemon")
    parser.add_argument("--latent_reference_multi_ip", action="store_true")
    parser.add_argument("--single_ip_condition", default="ip_doraemon")

    parser.add_argument("--adapter_type", choices=sorted(ADAPTER_TYPES), default="mlp")
    parser.add_argument("--adapter_depth", type=int, default=1)
    parser.add_argument("--bottleneck_dim", type=int, default=None)
    parser.add_argument("--adapter_attention_dim", type=int, default=None)
    parser.add_argument("--adapter_attention_heads", type=int, default=4)
    parser.add_argument("--adapter_attention_ffn_multiplier", type=int, default=4)
    parser.add_argument("--gate_type", choices=["none", "global", "token"], default="none")
    parser.add_argument(
        "--token_gate",
        action="store_true",
        help="--gate_type token 的快捷别名；和 --gate_type global 不能同时使用。",
    )
    parser.add_argument("--gate_init", type=float, default=0.5)
    parser.add_argument("--adapter_dropout", type=float, default=0.0)
    parser.add_argument("--residual_scale", type=float, default=0.5)
    parser.add_argument("--disable_risk_condition", action="store_true")
    parser.add_argument("--restrict_adapter_to_user_content_tokens", action="store_true")

    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=9)
    parser.add_argument("--t_min", type=int, default=4)
    parser.add_argument("--t_max", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--attention_backend", default=None)
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")

    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--benign_fraction", type=float, default=0.1)
    parser.add_argument("--max_unsafe_samples", type=int, default=None)
    parser.add_argument("--max_benign_samples", type=int, default=None)
    parser.add_argument("--filter_benign_label_safe", action="store_true")
    parser.add_argument("--extra_benign_csvs", default=None)
    parser.add_argument("--extra_benign_repeat", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--w_feedback", type=float, default=1.0)
    parser.add_argument(
        "--feedback_loss_type",
        "--dino_loss_type",
        choices=["cosine", "ranking"],
        default="cosine",
        help="cosine=directly minimize prototype similarity; ranking=make adapter score lower than original score.",
    )
    parser.add_argument("--ranking_margin", type=float, default=0.2)
    parser.add_argument("--w_emb", type=float, default=0.05)
    parser.add_argument("--w_id", type=float, default=0.2)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--torch_dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampler_seed", type=int, default=20260817)
    parser.add_argument("--fixed_train_seeds", action="store_true")
    parser.add_argument("--train_seed_upper_bound", type=int, default=2_147_483_647)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--reset_optimizer_on_resume", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_train_steps <= 0:
        raise ValueError("--max_train_steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive")
    if not 0.0 <= args.benign_fraction <= 1.0:
        raise ValueError("--benign_fraction must be in [0, 1]")
    if args.t_min < 1 or args.t_max < args.t_min or args.t_max > args.num_inference_steps:
        raise ValueError("invalid --t_min/--t_max range")
    if args.adapter_depth <= 0:
        raise ValueError("--adapter_depth must be positive")
    if args.adapter_attention_heads <= 0:
        raise ValueError("--adapter_attention_heads must be positive")
    if args.adapter_type == "bottleneck_self_attention":
        attention_dim = args.adapter_attention_dim or 256
        if attention_dim % args.adapter_attention_heads != 0:
            raise ValueError("--adapter_attention_dim must be divisible by --adapter_attention_heads")
    if args.adapter_type == "zimage_adaln_classifier_condition":
        if args.adapter_attention_dim and args.adapter_attention_dim % args.adapter_attention_heads != 0:
            raise ValueError("--adapter_attention_dim must be divisible by --adapter_attention_heads")
    if not 0.0 < args.gate_init < 1.0:
        raise ValueError("--gate_init must be in (0, 1)")
    if args.residual_scale <= 0:
        raise ValueError("--residual_scale must be positive")
    if args.token_gate and args.gate_type == "global":
        raise ValueError("--token_gate cannot be used together with --gate_type global")
    if args.ranking_margin < 0:
        raise ValueError("--ranking_margin must be non-negative")
    if args.resume_from_checkpoint and not Path(args.resume_from_checkpoint).is_file():
        raise FileNotFoundError(args.resume_from_checkpoint)
    if args.latent_reference_multi_ip and not args.latent_reference_features:
        raise ValueError("--latent_reference_multi_ip requires --latent_reference_features")


def resolve_risk_ids(
    samples: Sequence[PromptSample],
    rng: random.Random,
    *,
    multi_ip: bool,
    single_ip_condition: str,
) -> list[str]:
    risk_ids: list[str] = []
    for sample in samples:
        if sample.is_benign:
            risk_ids.append(rng.choice(IP_CONDITION_NAMES))
            continue
        if not multi_ip:
            risk_ids.append(single_ip_condition)
            continue
        metadata = dict(sample.metadata)
        metadata["sample_id"] = sample.sample_id
        condition = ip_condition_from_metadata(metadata, fallback="")
        if condition not in IP_CONDITION_NAMES:
            raise ValueError(
                "无法从 unsafe sample metadata 推断 IP condition: "
                f"sample_id={sample.sample_id!r}, metadata={metadata}"
            )
        risk_ids.append(condition)
    return risk_ids


def mean_embedding_mse(
    original_embeds: Sequence[torch.Tensor],
    safe_embeds: Sequence[torch.Tensor],
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    values = []
    for index, (original, safe) in enumerate(zip(original_embeds, safe_embeds)):
        if sample_mask is not None and not bool(sample_mask[index]):
            continue
        values.append((safe.float() - original.float()).pow(2).mean())
    if not values:
        return safe_embeds[0].float().sum() * 0.0
    return torch.stack(values).mean()


def sample_latent_seeds(
    samples: Sequence[PromptSample],
    rng: random.Random,
    *,
    fixed: bool,
    upper_bound: int,
) -> list[int]:
    if fixed:
        return [int(sample.seed) for sample in samples]
    return [rng.randrange(0, upper_bound) for _ in samples]


def build_adapter_config(args: argparse.Namespace, embedding_dim: int) -> dict:
    return {
        "adapter_type": args.adapter_type,
        "embedding_dim": int(embedding_dim),
        "bottleneck_dim": args.bottleneck_dim,
        "attention_dim": args.adapter_attention_dim,
        "attention_heads": args.adapter_attention_heads,
        "attention_ffn_multiplier": args.adapter_attention_ffn_multiplier,
        "adapter_depth": args.adapter_depth,
        "gate_type": args.gate_type,
        "learnable_gate": args.gate_type != "none",
        "gate_init": args.gate_init,
        "residual_scale": args.residual_scale,
        "dropout": args.adapter_dropout,
        "use_risk_condition": not args.disable_risk_condition,
        "num_risk_types": 9,
        "num_classifier_classes": 5,
        "zero_init_depth2": True,
        "clamp_delta": True,
    }


def save_checkpoint(
    accelerator: Accelerator,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    args: argparse.Namespace,
    adapter_config: dict,
    embedding_dim: int,
) -> None:
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "adapter_state_dict": accelerator.unwrap_model(adapter).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "embedding_dim": int(embedding_dim),
        "train_type": "latent_backbone_prompt_embedding_adapter",
        "adapter_config": dict(adapter_config),
        "proxy_config": {
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "t_min": args.t_min,
            "t_max": args.t_max,
            "guidance_scale": args.guidance_scale,
            "max_sequence_length": args.max_sequence_length,
        },
        "train_config": {
            "feedback_model": "latent_backbone",
            "feedback_loss_type": args.feedback_loss_type,
            "ranking_margin": float(args.ranking_margin),
            "target_risk": "ip",
            "latent_reference_features": args.latent_reference_features,
            "latent_reference_key": args.latent_reference_key,
            "latent_reference_multi_ip": bool(args.latent_reference_multi_ip),
            "latent_backbone_checkpoint": args.latent_backbone_checkpoint,
            "model_path": args.model_path,
            "unsafe_csv": args.unsafe_csv,
            "benign_csv": args.benign_csv,
        },
        "args": vars(args),
    }
    accelerator.save(payload, checkpoint_dir / f"step{step:06d}.pt")
    accelerator.save(payload, checkpoint_dir / "latest.pt")
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.token_gate:
        args.gate_type = "token"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        cpu=str(args.device).lower() == "cpu",
    )
    device = accelerator.device
    dtype = torch_dtype_from_name(args.torch_dtype)
    process_seed = args.sampler_seed + accelerator.process_index * 100003
    random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)

    dataset_config = DatasetConfig(
        unsafe_csv=args.unsafe_csv,
        benign_csv=args.benign_csv,
        extra_benign_csvs=args.extra_benign_csvs,
        extra_benign_repeat=args.extra_benign_repeat,
        filter_benign_label_safe=args.filter_benign_label_safe,
        max_unsafe_samples=args.max_unsafe_samples,
        max_benign_samples=args.max_benign_samples,
        sample_seed_base=args.sampler_seed,
    )
    unsafe_samples, benign_samples = load_prompt_samples(dataset_config)
    train_samples = split_prompt_samples(unsafe_samples, benign_samples, dataset_config)["train"]
    accelerator.print(
        f"[minimal-train] unsafe={len(unsafe_samples)} benign={len(benign_samples)} "
        f"train={len(train_samples)}"
    )

    proxy_config = ProxyLatentConfig(
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        t_min=args.t_min,
        t_max=args.t_max,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
    )
    pipe = ZImageTrainPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.to(device)
    freeze_module(pipe.text_encoder)
    freeze_module(pipe.transformer)
    freeze_module(pipe.vae)
    if args.attention_backend:
        pipe.transformer.set_attention_backend(args.attention_backend)
    if not args.disable_gradient_checkpointing and hasattr(
        pipe.transformer,
        "enable_gradient_checkpointing",
    ):
        pipe.transformer.enable_gradient_checkpointing()

    proxy_runner = ZImageProxyLatentRunner(pipe, proxy_config)
    first_embeds, _ = proxy_runner.encode_prompts([train_samples[0].prompt])
    embedding_dim = int(first_embeds[0].shape[-1])
    adapter_config = build_adapter_config(args, embedding_dim)

    condition_embeddings = None
    if args.adapter_type == "zimage_adaln_text_condition":
        condition_embeddings = precompute_ip_condition_embeddings(
            proxy_runner,
            num_risk_types=9,
            embedding_dim=embedding_dim,
        )
    adapter = build_adapter_from_config(
        adapter_config,
        embedding_dim=embedding_dim,
        condition_embeddings=condition_embeddings,
    ).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    adapter, optimizer = accelerator.prepare(adapter, optimizer)

    latent_encoder = LatentBackboneFeatureEncoder(
        args.latent_backbone_checkpoint,
        config_path=args.latent_backbone_config,
        model_file=args.latent_backbone_model_file,
        device=device,
        torch_dtype=torch_dtype_from_name(args.latent_backbone_torch_dtype),
        input_size=args.latent_backbone_input_size,
    )
    if args.latent_reference_multi_ip:
        prototypes, reference_metadata = load_latent_reference_prototypes(
            args.latent_reference_features,
        )
        scorer = LatentBackboneCosineScorer(
            encoder=latent_encoder,
            target_prototypes=prototypes,
            default_target_key=args.latent_reference_key,
            reference_metadata=reference_metadata,
        ).to(device)
        accelerator.print(
            f"[minimal-train] multi-IP keys={reference_metadata['reference_keys']}"
        )
    else:
        prototype, reference_metadata = load_latent_reference_payload(
            args.latent_reference_features,
            target_key=args.latent_reference_key,
        )
        scorer = LatentBackboneCosineScorer(
            encoder=latent_encoder,
            target_prototype=prototype,
            reference_metadata=reference_metadata,
        ).to(device)
    scorer.eval()

    start_step = 0
    if args.resume_from_checkpoint:
        payload = torch.load(args.resume_from_checkpoint, map_location="cpu", weights_only=True)
        accelerator.unwrap_model(adapter).load_state_dict(
            payload["adapter_state_dict"],
            strict=True,
        )
        if not args.reset_optimizer_on_resume and payload.get("optimizer_state_dict"):
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = int(payload.get("step", 0))
        accelerator.print(f"[minimal-train] resumed step={start_step}")

    sampler = BalancedPromptBatchSampler(
        train_samples,
        batch_size=args.batch_size,
        benign_fraction=args.benign_fraction,
        seed=process_seed,
    )
    condition_rng = random.Random(process_seed + 2017)
    target_step_rng = random.Random(process_seed + 101)
    latent_seed_rng = random.Random(process_seed + 1009)
    log_path = Path(args.output_dir) / "train_log.jsonl"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_start = time.time()

    for step in range(start_step + 1, args.max_train_steps + 1):
        batch = sampler.sample_batch()
        benign_mask = batch.benign_mask(device)
        risk_ids = resolve_risk_ids(
            batch.samples,
            condition_rng,
            multi_ip=args.latent_reference_multi_ip,
            single_ip_condition=args.single_ip_condition,
        )
        with accelerator.accumulate(adapter):
            if args.restrict_adapter_to_user_content_tokens:
                original_embeds, negative_embeds, token_masks = (
                    proxy_runner.encode_prompts_with_content_token_masks(batch.prompts)
                )
                token_masks = [mask.to(device) for mask in token_masks]
            else:
                original_embeds, negative_embeds = proxy_runner.encode_prompts(batch.prompts)
                token_masks = None
            original_embeds = [embed.to(device) for embed in original_embeds]
            negative_embeds = [embed.to(device) for embed in negative_embeds]
            safe_embeds = adapter(
                original_embeds,
                risk_ids=risk_ids,
                token_masks=token_masks,
            )
            target_step = target_step_rng.randint(args.t_min, args.t_max)
            latent_seeds = sample_latent_seeds(
                batch.samples,
                latent_seed_rng,
                fixed=args.fixed_train_seeds,
                upper_bound=args.train_seed_upper_bound,
            )
            latent_x1 = proxy_runner.denoise_to_step(
                safe_embeds,
                negative_prompt_embeds=negative_embeds,
                target_step=target_step,
                seeds=latent_seeds,
            )
            if args.latent_reference_multi_ip:
                scores = scorer(latent_x1, target_keys=risk_ids)
            else:
                scores = scorer(latent_x1)
            unsafe_mask = ~benign_mask
            unsafe_scores = scores[unsafe_mask]
            original_scores = None
            if args.feedback_loss_type == "ranking":
                with torch.no_grad():
                    original_latent_x1 = proxy_runner.denoise_to_step(
                        original_embeds,
                        negative_prompt_embeds=negative_embeds,
                        target_step=target_step,
                        seeds=latent_seeds,
                    )
                    if args.latent_reference_multi_ip:
                        original_scores = scorer(original_latent_x1, target_keys=risk_ids)
                    else:
                        original_scores = scorer(original_latent_x1)
                unsafe_original_scores = original_scores[unsafe_mask]
                loss_feedback = (
                    F.softplus(
                        unsafe_scores.float()
                        - unsafe_original_scores.float()
                        + float(args.ranking_margin)
                    ).mean()
                    if unsafe_scores.numel()
                    else scores.sum() * 0.0
                )
            else:
                loss_feedback = (
                    (1.0 + unsafe_scores.float()).mean()
                    if unsafe_scores.numel()
                    else scores.sum() * 0.0
                )
            loss_emb = mean_embedding_mse(original_embeds, safe_embeds)
            loss_id = mean_embedding_mse(
                original_embeds,
                safe_embeds,
                sample_mask=benign_mask,
            )
            loss_total = (
                args.w_feedback * loss_feedback
                + args.w_emb * loss_emb
                + args.w_id * loss_id
            )
            accelerator.backward(loss_total)
            if accelerator.sync_gradients:
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(adapter.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == 1:
            metrics = {
                "step": step,
                "loss_total": float(loss_total.detach().cpu()),
                "loss_feedback": float(loss_feedback.detach().cpu()),
                "loss_embedding": float(loss_emb.detach().cpu()),
                "loss_identity": float(loss_id.detach().cpu()),
                "score_mean": float(scores.detach().mean().cpu()),
                "score_unsafe": (
                    float(scores.detach()[~benign_mask].mean().cpu())
                    if bool((~benign_mask).any())
                    else None
                ),
                "original_score_unsafe": (
                    float(original_scores.detach()[~benign_mask].mean().cpu())
                    if original_scores is not None and bool((~benign_mask).any())
                    else None
                ),
                "feedback_loss_type": args.feedback_loss_type,
                "ranking_margin": float(args.ranking_margin),
                "target_step": target_step,
                "risk_ids": risk_ids,
                "latent_seeds": latent_seeds,
                "elapsed_sec": round(time.time() - train_start, 3),
            }
            if accelerator.is_main_process:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                accelerator.print(
                    f"[minimal-train] step={step} loss={metrics['loss_total']:.6f} "
                    f"score={metrics['score_mean']:.6f} target_step={target_step}"
                )

        if step % args.save_every == 0 or step == args.max_train_steps:
            save_checkpoint(
                accelerator,
                adapter,
                optimizer,
                step=step,
                args=args,
                adapter_config=adapter_config,
                embedding_dim=embedding_dim,
            )

    accelerator.print("[minimal-train] finished")
    accelerator.end_training()


if __name__ == "__main__":
    main()
