#!/usr/bin/env python
"""Minimal base-vs-adapter generation tester."""

from __future__ import annotations

import argparse
import json
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
from safe_embedding_adapter.z03 import Z03Scorer
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
    parser.add_argument(
        "--z03_ckpt",
        default=None,
        help="可选 Z-03 分类头 checkpoint；提供后会额外保存每张图的风险概率和 safe margin。",
    )
    parser.add_argument("--z03_model_type", choices=["auto", "legacy", "export"], default="auto")
    parser.add_argument("--z03_model_file", default=None)
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


def get_prompt_tokens(
    pipe: ZImageTrainPipeline,
    prompt: str,
    *,
    max_sequence_length: int,
) -> tuple[list[int], list[str]]:
    """Return the exact valid token sequence used by the Z-Image text encoder."""

    formatted_prompt = pipe._format_user_prompt(prompt)
    text_inputs = pipe.tokenizer(
        [formatted_prompt],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    valid_mask = text_inputs.attention_mask[0].bool()
    token_ids = text_inputs.input_ids[0][valid_mask].tolist()
    tokens = pipe.tokenizer.convert_ids_to_tokens(token_ids)
    return [int(token_id) for token_id in token_ids], [str(token) for token in tokens]


def get_last_token_gate_values(adapter: torch.nn.Module) -> list[list[float]] | None:
    """Read per-block token gates from the most recent single-sample forward."""

    structured = getattr(adapter, "_last_token_gate_tensors", None)
    if getattr(adapter, "gate_type", None) == "token" and structured:
        sample_gates = structured[0]
        if sample_gates:
            return [
                [float(value) for value in tensor.detach().cpu().flatten().tolist()]
                for tensor in sample_gates
            ]

    # Attention-based adapters keep a flat list. The test script forwards one
    # sample at a time, so each item corresponds to one adapter block.
    flat = getattr(adapter, "_last_gate_tensors", None)
    if flat:
        return [
            [float(value) for value in tensor.detach().cpu().flatten().tolist()]
            for tensor in flat
        ]
    return None


def build_token_gate_record(
    *,
    adapter: torch.nn.Module,
    pipe: ZImageTrainPipeline,
    sample: PromptSample,
    condition: str,
    token_masks: Sequence[torch.Tensor] | None,
    max_sequence_length: int,
    index: int,
) -> dict | None:
    gate_values_by_block = get_last_token_gate_values(adapter)
    if not gate_values_by_block:
        return None

    token_ids, tokens = get_prompt_tokens(
        pipe,
        sample.prompt,
        max_sequence_length=max_sequence_length,
    )
    token_count = len(tokens)
    content_mask = None
    if token_masks is not None:
        content_mask = [bool(value) for value in token_masks[0].detach().cpu().tolist()]
        if len(content_mask) != token_count:
            raise ValueError(
                "token mask 长度和 tokenizer 有效 token 数不一致："
                f"mask={len(content_mask)}, tokens={token_count}"
            )

    gate_length = len(gate_values_by_block[0])
    if any(len(values) != gate_length for values in gate_values_by_block):
        raise ValueError("不同 token-gate block 的 token 数量不一致")

    if content_mask is None:
        editable_indices = list(range(token_count))
    else:
        editable_indices = [idx for idx, editable in enumerate(content_mask) if editable]

    if gate_length == token_count:
        gate_token_indices = list(range(token_count))
    elif gate_length == len(editable_indices):
        gate_token_indices = editable_indices
    else:
        raise ValueError(
            "token gate 数量和 prompt token 数不匹配："
            f"gate={gate_length}, tokens={token_count}, editable={len(editable_indices)}"
        )

    token_to_gate_position = {
        token_index: gate_index
        for gate_index, token_index in enumerate(gate_token_indices)
    }
    token_records = []
    for token_index, (token_id, token) in enumerate(zip(token_ids, tokens)):
        gate_position = token_to_gate_position.get(token_index)
        if gate_position is None:
            gates = [None for _ in gate_values_by_block]
            gate_mean = None
            editable = False
        else:
            gates = [
                round(values[gate_position], 8)
                for values in gate_values_by_block
            ]
            gate_mean = round(sum(gates) / len(gates), 8)
            editable = True
        token_records.append(
            {
                "index": token_index,
                "token_id": int(token_id),
                "token": token,
                "editable": editable,
                "gates": gates,
                "gate_mean": gate_mean,
            }
        )

    return {
        "index": index,
        "sample_id": sample.sample_id,
        "prompt": sample.prompt,
        "condition": condition,
        "adapter_type": getattr(adapter, "adapter_type", adapter.__class__.__name__),
        "gate_type": getattr(adapter, "gate_type", "token"),
        "num_blocks": len(gate_values_by_block),
        "block_gate_means": [
            round(sum(values) / len(values), 8)
            for values in gate_values_by_block
        ],
        "tokens": token_records,
    }


def print_token_gate_record(record: dict) -> None:
    print("[minimal-test][token-gate] prompt:")
    print(record["prompt"])
    print(
        f"[minimal-test][token-gate] sample={record['sample_id']} "
        f"condition={record['condition']} blocks={record['num_blocks']}"
    )
    block_labels = ", ".join(f"block_{index}" for index in range(record["num_blocks"]))
    print(f"[minimal-test][token-gate] gate_columns=[{block_labels}]")
    for token_record in record["tokens"]:
        gates = token_record["gates"]
        gate_text = (
            "[" + ", ".join("None" if value is None else f"{value:.6f}" for value in gates) + "]"
        )
        print(
            f"  token[{token_record['index']:03d}] "
            f"id={token_record['token_id']} "
            f"text={token_record['token']!r} "
            f"editable={token_record['editable']} "
            f"gate={gate_text}"
        )


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
    return_latent: bool = False,
) -> Image.Image | tuple[Image.Image, torch.Tensor]:
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

    # Z-03 expects the raw denoising output latent_x1. Decode a separate
    # VAE-scaled copy so scoring and visualization use the correct tensors.
    raw_latents = latents.float().detach() if return_latent else None
    vae_latents = latents.to(pipe.vae.dtype)
    vae_latents = vae_latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(vae_latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    if return_latent:
        return image, raw_latents
    return image


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
    train_config = checkpoint.get("train_config") or {}
    z03_ckpt = args.z03_ckpt or train_config.get("z03_ckpt")
    z03_scorer = None
    if z03_ckpt:
        z03_scorer = Z03Scorer(
            z03_ckpt,
            device=device,
            model_type=args.z03_model_type,
            model_file=args.z03_model_file or train_config.get("z03_model_file"),
        ).to(device)
        z03_scorer.eval()
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
    score_handle = None
    token_gate_handle = None
    if z03_scorer is not None:
        score_path = output_root / "z03_scores.jsonl"
        score_handle = score_path.open("w", encoding="utf-8")

    try:
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
            token_gate_record = build_token_gate_record(
                adapter=adapter,
                pipe=pipe,
                sample=sample,
                condition=condition,
                token_masks=token_masks,
                max_sequence_length=args.max_sequence_length,
                index=index,
            )
            if token_gate_record is not None:
                if token_gate_handle is None:
                    token_gate_handle = (
                        output_root / "token_gates.jsonl"
                    ).open("w", encoding="utf-8")
                print_token_gate_record(token_gate_record)
                token_gate_handle.write(
                    json.dumps(token_gate_record, ensure_ascii=False) + "\n"
                )
                token_gate_handle.flush()

            seed = args.generation_seed + index
            adapted_result = generate_image(
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
                return_latent=z03_scorer is not None,
            )
            if z03_scorer is not None:
                adapted_image, adapted_latent = adapted_result
            else:
                adapted_image = adapted_result
                adapted_latent = None
            name = f"{index:04d}_{safe_filename(sample.sample_id)}"
            adapted_image.save(adapter_dir / f"{name}_adapter.png")
            base_image = None
            base_latent = None
            if not args.skip_base_generation:
                base_result = generate_image(
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
                    return_latent=z03_scorer is not None,
                )
                if z03_scorer is not None:
                    base_image, base_latent = base_result
                else:
                    base_image = base_result
                base_image.save(base_dir / f"{name}_base.png")
                save_comparison(base_image, adapted_image, compare_dir / f"{name}_compare.png")
            if z03_scorer is not None:
                with torch.no_grad():
                    adapted_output = z03_scorer(adapted_latent)
                    adapted_prob = float(adapted_output.risk_prob(target_risk)[0].cpu())
                    adapted_margin = float(adapted_output.risk_safe_margin(target_risk)[0].cpu())
                    score_record = {
                        "index": index,
                        "sample_id": sample.sample_id,
                        "prompt": sample.prompt,
                        "target_risk": target_risk,
                        "seed": seed,
                        "adapter_risk_prob": adapted_prob,
                        "adapter_safe_margin": adapted_margin,
                    }
                    if base_latent is not None:
                        base_output = z03_scorer(base_latent)
                        score_record["base_risk_prob"] = float(base_output.risk_prob(target_risk)[0].cpu())
                        score_record["base_safe_margin"] = float(base_output.risk_safe_margin(target_risk)[0].cpu())
                score_handle.write(json.dumps(score_record, ensure_ascii=False) + "\n")
                score_handle.flush()
            print(f"[minimal-test] {index + 1}/{len(samples)} condition={condition} seed={seed}")
    finally:
        if score_handle is not None:
            score_handle.close()
        if token_gate_handle is not None:
            token_gate_handle.close()

    print(f"[minimal-test] saved outputs to {output_root}")


if __name__ == "__main__":
    main()
