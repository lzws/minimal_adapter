#!/usr/bin/env python
"""Minimal base-vs-adapter generation tester."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

from safe_embedding_adapter.adapter_factory import (
    build_adapter_from_config,
    infer_adapter_type_from_state_dict,
    infer_embedding_dim_from_state_dict,
    infer_gate_type_from_state_dict,
    normalize_adapter_state_dict,
)
from safe_embedding_adapter.config import DatasetConfig, torch_dtype_from_name
from safe_embedding_adapter.data import PromptSample, load_prompt_samples, split_prompt_samples
from safe_embedding_adapter.model import (
    IP_CONDITION_NAMES,
    adapter_supports_risk_id,
    freeze_module,
    ip_condition_from_metadata,
)
from safe_embedding_adapter.zimage_proxy import (
    calculate_shift,
    get_default_z_image_sigmas,
    retrieve_timesteps,
)
from safe_embedding_adapter.zimage_train_pipeline import ZImageTrainPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate base and adapter images for any adapter checkpoint.")
    parser.add_argument("--adapter_ckpt", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--unsafe_csv", required=True)
    parser.add_argument("--benign_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_set", choices=["unsafe", "benign", "mixed"], default="unsafe")
    parser.add_argument("--target_risk", default=None)
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--sample_seed", type=int, default=20260817)
    parser.add_argument("--generation_seed", type=int, default=42)
    parser.add_argument("--use_all_data", action="store_true")
    parser.add_argument(
        "--skip_base_generation",
        action="store_true",
        help="Only generate adapter images; skip base and side-by-side compare images.",
    )
    parser.add_argument("--filter_metadata_key", default=None)
    parser.add_argument("--filter_metadata_value", default=None)
    parser.add_argument("--group_by_metadata_key", default=None)
    parser.add_argument("--group_metadata_values", default=None)
    parser.add_argument("--samples_per_group", type=int, default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=9)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--cfg_normalization", type=float, default=0.0)
    parser.add_argument("--cfg_truncation", type=float, default=1.0)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--restrict_adapter_to_user_content_tokens", action="store_true")
    parser.add_argument("--torch_dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention_backend", default=None)
    return parser.parse_args()


def safe_filename(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    return value[:80] or "sample"


def parse_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def pick_samples(args: argparse.Namespace) -> list[PromptSample]:
    config = DatasetConfig(unsafe_csv=args.unsafe_csv, benign_csv=args.benign_csv)
    unsafe_samples, benign_samples = load_prompt_samples(config)
    if args.use_all_data:
        unsafe_pool = list(unsafe_samples)
        benign_pool = list(benign_samples)
    else:
        splits = split_prompt_samples(unsafe_samples, benign_samples, config)
        unsafe_pool = splits["test_unsafe"]
        benign_pool = splits["test_benign"]

    if args.prompt_set == "unsafe":
        pool = unsafe_pool
    elif args.prompt_set == "benign":
        pool = benign_pool
    else:
        rng = random.Random(args.sample_seed)
        unsafe_count = args.num_prompts // 2
        unsafe_pick = rng.sample(unsafe_pool, min(unsafe_count, len(unsafe_pool)))
        benign_pick = rng.sample(
            benign_pool,
            min(args.num_prompts - len(unsafe_pick), len(benign_pool)),
        )
        pool = unsafe_pick + benign_pick
        rng.shuffle(pool)

    if args.filter_metadata_key:
        if args.filter_metadata_value is None:
            raise ValueError("--filter_metadata_key requires --filter_metadata_value")
        pool = [
            sample
            for sample in pool
            if str(sample.metadata.get(args.filter_metadata_key, "")).strip().casefold()
            == str(args.filter_metadata_value).strip().casefold()
        ]

    if args.group_by_metadata_key:
        values = parse_values(args.group_metadata_values)
        if not values or not args.samples_per_group:
            raise ValueError("group filtering requires --group_metadata_values and --samples_per_group")
        rng = random.Random(args.sample_seed)
        grouped: list[PromptSample] = []
        for value in values:
            group = [
                sample
                for sample in pool
                if str(sample.metadata.get(args.group_by_metadata_key, "")).strip().casefold()
                == value.casefold()
            ]
            if len(group) < args.samples_per_group:
                raise ValueError(
                    f"group {args.group_by_metadata_key}={value!r} has only {len(group)} samples"
                )
            grouped.extend(rng.sample(group, args.samples_per_group))
        pool = grouped

    if not pool:
        raise ValueError("no samples available for testing")
    if len(pool) > args.num_prompts and not args.group_by_metadata_key:
        pool = random.Random(args.sample_seed).sample(pool, args.num_prompts)
    return pool


def load_adapter(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = normalize_adapter_state_dict(checkpoint["adapter_state_dict"])
    config = dict(checkpoint.get("adapter_config") or {})
    embedding_dim = int(config.get("embedding_dim") or infer_embedding_dim_from_state_dict(state_dict))
    config["embedding_dim"] = embedding_dim
    config.setdefault("adapter_type", infer_adapter_type_from_state_dict(state_dict))
    config.setdefault("gate_type", infer_gate_type_from_state_dict(state_dict))
    config.setdefault("num_risk_types", 9)
    condition_embeddings = state_dict.get("condition_embeddings")
    adapter = build_adapter_from_config(
        config,
        embedding_dim=embedding_dim,
        condition_embeddings=condition_embeddings,
    )
    adapter.load_state_dict(state_dict, strict=True)
    adapter.to(device)
    adapter.eval()
    return adapter, checkpoint


def risk_condition_for_sample(
    adapter: torch.nn.Module,
    sample: PromptSample,
    target_risk: str,
) -> str:
    if target_risk != "ip":
        return target_risk
    if sample.is_benign:
        selector = sum(str(sample.sample_id).encode("utf-8")) % len(IP_CONDITION_NAMES)
        return IP_CONDITION_NAMES[selector]
    metadata = dict(sample.metadata)
    metadata["sample_id"] = sample.sample_id
    condition = ip_condition_from_metadata(metadata, fallback="")
    if condition in IP_CONDITION_NAMES and adapter_supports_risk_id(adapter, condition):
        return condition
    if getattr(adapter, "adapter_type", "") == "zimage_adaln_classifier_condition":
        raise ValueError(f"cannot infer IP condition for sample {sample.sample_id!r}")
    return "ip"


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def prepare_timesteps(pipe, latents: torch.Tensor, steps: int, device: torch.device):
    image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, steps = retrieve_timesteps(
        pipe.scheduler,
        steps,
        device,
        sigmas=get_default_z_image_sigmas(steps),
        mu=mu,
    )
    pipe._num_timesteps = len(timesteps)
    pipe.scheduler.set_begin_index(0)
    return timesteps


@torch.no_grad()
def generate_image(
    pipe,
    prompt_embeds: Sequence[torch.Tensor],
    negative_prompt_embeds: Sequence[torch.Tensor],
    *,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    cfg_normalization: float,
    cfg_truncation: float,
    generator: torch.Generator,
) -> Image.Image:
    device = torch.device(pipe._execution_device)
    latents = pipe.prepare_latents(
        1,
        pipe.transformer.in_channels,
        height,
        width,
        torch.float32,
        device,
        generator,
        None,
    )
    timesteps = prepare_timesteps(pipe, latents, steps, device)
    do_cfg = guidance_scale > 0
    if do_cfg and not negative_prompt_embeds:
        raise ValueError("guidance_scale > 0 requires negative_prompt_embeds")

    if do_cfg and cfg_truncation <= 1:
        normalized_timesteps = ((1000 - timesteps.float()) / 1000).tolist()
    else:
        normalized_timesteps = None

    for index, timestep_raw in enumerate(timesteps):
        timestep = timestep_raw.expand(1)
        timestep_normalized = (1000 - timestep) / 1000
        current_scale = guidance_scale
        if normalized_timesteps is not None and normalized_timesteps[index] > cfg_truncation:
            current_scale = 0.0
        apply_cfg = do_cfg and current_scale > 0

        if apply_cfg:
            latent_input = latents.to(pipe.transformer.dtype).repeat(2, 1, 1, 1)
            prompt_input = list(prompt_embeds) + list(negative_prompt_embeds)
            timestep_input = timestep_normalized.repeat(2)
        else:
            latent_input = latents.to(pipe.transformer.dtype)
            prompt_input = list(prompt_embeds)
            timestep_input = timestep_normalized

        model_output_list = pipe.transformer(
            list(latent_input.unsqueeze(2).unbind(dim=0)),
            timestep_input,
            prompt_input,
            return_dict=False,
        )[0]
        if apply_cfg:
            positive = model_output_list[0].float()
            negative = model_output_list[1].float()
            velocity = positive + current_scale * (positive - negative)
            if cfg_normalization > 0:
                positive_norm = torch.linalg.vector_norm(positive)
                velocity_norm = torch.linalg.vector_norm(velocity)
                max_norm = positive_norm * cfg_normalization
                if velocity_norm > max_norm:
                    velocity = velocity * (max_norm / velocity_norm)
            velocity = velocity.unsqueeze(0)
        else:
            velocity = torch.stack(
                [item.float() for item in model_output_list],
                dim=0,
            )

        velocity = -velocity.squeeze(2)
        latents = pipe.scheduler.step(
            velocity,
            timestep_raw,
            latents,
            return_dict=False,
        )[0].to(torch.float32)

    latents = latents.to(pipe.vae.dtype)
    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def save_comparison(base: Image.Image, adapted: Image.Image, path: Path) -> None:
    width, height = base.size
    canvas = Image.new("RGB", (width * 2, height + 32), color="white")
    canvas.paste(base.convert("RGB"), (0, 32))
    canvas.paste(adapted.convert("RGB"), (width, 32))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), "base", fill="black", font=font)
    draw.text((width + 8, 8), "adapter", fill="black", font=font)
    canvas.save(path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch_dtype_from_name(args.torch_dtype)
    samples = pick_samples(args)
    adapter, checkpoint = load_adapter(args.adapter_ckpt, device)
    target_risk = args.target_risk or str(
        (checkpoint.get("train_config") or {}).get("target_risk", "ip")
    )
    pipe = ZImageTrainPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.to(device)
    freeze_module(pipe.text_encoder)
    freeze_module(pipe.transformer)
    freeze_module(pipe.vae)
    if args.attention_backend:
        pipe.transformer.set_attention_backend(args.attention_backend)

    output_root = Path(args.output_dir)
    base_dir = output_root / "base"
    adapter_dir = output_root / "adapter"
    compare_dir = output_root / "compare"
    for directory in (base_dir, adapter_dir, compare_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples):
        condition = risk_condition_for_sample(adapter, sample, target_risk)
        if args.restrict_adapter_to_user_content_tokens:
            prompt_embeds, negative_embeds, token_masks = (
                pipe.encode_prompt_with_content_token_masks(
                    [sample.prompt],
                    device=device,
                    do_classifier_free_guidance=args.guidance_scale > 0,
                    max_sequence_length=args.max_sequence_length,
                )
            )
            token_masks = [mask.to(device) for mask in token_masks]
        else:
            prompt_embeds, negative_embeds = pipe.encode_prompt(
                [sample.prompt],
                device=device,
                do_classifier_free_guidance=args.guidance_scale > 0,
                max_sequence_length=args.max_sequence_length,
            )
            token_masks = None
        prompt_embeds = [embed.to(device) for embed in prompt_embeds]
        negative_embeds = [embed.to(device) for embed in negative_embeds]
        with torch.no_grad():
            adapted_embeds = adapter(
                prompt_embeds,
                risk_ids=[condition],
                token_masks=token_masks,
            )

        seed = args.generation_seed + index
        adapted_image = generate_image(
            pipe,
            adapted_embeds,
            negative_embeds,
            height=args.height,
            width=args.width,
            steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            cfg_normalization=args.cfg_normalization,
            cfg_truncation=args.cfg_truncation,
            generator=make_generator(device, seed),
        )
        name = f"{index:04d}_{safe_filename(sample.sample_id)}"
        adapted_image.save(adapter_dir / f"{name}_adapter.png")
        if not args.skip_base_generation:
            base_image = generate_image(
                pipe,
                prompt_embeds,
                negative_embeds,
                height=args.height,
                width=args.width,
                steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                cfg_normalization=args.cfg_normalization,
                cfg_truncation=args.cfg_truncation,
                generator=make_generator(device, seed),
            )
            base_image.save(base_dir / f"{name}_base.png")
            save_comparison(base_image, adapted_image, compare_dir / f"{name}_compare.png")
        print(f"[minimal-test] {index + 1}/{len(samples)} condition={condition} seed={seed}")

    print(f"[minimal-test] saved outputs to {output_root}")


if __name__ == "__main__":
    main()
