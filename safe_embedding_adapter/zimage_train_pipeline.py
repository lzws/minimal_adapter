"""训练用最小 Z-Image pipeline。

仓库里的推理 pipeline 继承了 diffusers 的 LoRA mixin：
    FromSingleFileMixin, ZImageLoraLoaderMixin

adapter 训练并不需要 LoRA 加载、单文件加载或图像后处理；只需要：
    - from_pretrained 加载 scheduler/vae/text_encoder/tokenizer/transformer；
    - encode_prompt 得到变长 prompt embeddings；
    - prepare_latents 生成初始噪声 latent。

因此这里定义一个最小 pipeline，专门服务可微 latent_x1 proxy 训练。
"""

from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoTokenizer, PreTrainedModel

from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import ZImageTransformer2DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor


class ZImageTrainPipeline(DiffusionPipeline):
    """只保留 adapter 训练需要的 Z-Image pipeline 能力。

    这个类现在也提供一个精简版 __call__，但它只返回 latent_x1，不负责 decode
    成图像。训练代码实际使用的是 ZImageProxyLatentRunner；__call__ 只是把
    同一个 runner 暴露到 pipeline 侧，方便人工检查和避免误解。
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: PreTrainedModel,
        tokenizer: AutoTokenizer,
        transformer: ZImageTransformer2DModel,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        # key=max_sequence_length，value=(prefix_token_count, suffix_token_count)。
        # Z-Image 的 user prompt 被固定 chat template 包住，模板 token 只在
        # user content 的前后出现，因此 content mask 不需要每条 prompt 都用
        # offset_mapping 重算。第一次用 marker 定位边界，后续直接复用首尾 token 数。
        self._content_token_boundary_cache: dict[int, tuple[int, int]] = {}

    def _format_user_prompt(self, prompt_item: str) -> str:
        """按 Z-Image chat template 格式化单条 user prompt。"""

        messages = [{"role": "user", "content": prompt_item}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def _content_token_boundary_counts(self, max_sequence_length: int) -> tuple[int, int]:
        """计算固定 chat template 在 user content 前后的有效 token 数。

        Returns:
            prefix_token_count:
                完整有效 token 序列里，user content 第一个 token 之前的 token 数。
            suffix_token_count:
                完整有效 token 序列里，user content 最后一个 token 之后的 token 数。

        这个函数只在第一次遇到某个 max_sequence_length 时调用一次。它仍使用
        tokenizer offset_mapping，但只用于定位固定模板边界，不再对每条训练
        prompt 重复做字符 span 匹配。
        """

        cache_key = int(max_sequence_length)
        if cache_key in self._content_token_boundary_cache:
            return self._content_token_boundary_cache[cache_key]

        marker = "SAFE_EMBEDDING_ADAPTER_USER_CONTENT_MARKER_20260730"
        formatted_prompt = self._format_user_prompt(marker)
        marker_start = formatted_prompt.find(marker)
        if marker_start < 0:
            raise ValueError("无法在 chat template 后的 marker prompt 中定位 user content")
        marker_end = marker_start + len(marker)

        text_inputs = self.tokenizer(
            [formatted_prompt],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = getattr(text_inputs, "offset_mapping", None)
        if offset_mapping is None:
            raise ValueError("当前 tokenizer 未返回 offset_mapping，无法初始化 user content token 边界")

        attention_mask = text_inputs.attention_mask[0].bool()
        valid_offsets = offset_mapping[0][attention_mask]
        content_indices = []
        for token_index, (token_start, token_end) in enumerate(valid_offsets.tolist()):
            token_start = int(token_start)
            token_end = int(token_end)
            overlaps_marker = max(token_start, marker_start) < min(token_end, marker_end)
            if overlaps_marker:
                content_indices.append(token_index)

        if not content_indices:
            raise ValueError(
                "初始化 user content token 边界失败：marker 没有对应有效 token，"
                f"max_sequence_length={max_sequence_length}"
            )

        prefix_token_count = int(content_indices[0])
        suffix_token_count = int(valid_offsets.shape[0]) - int(content_indices[-1]) - 1
        self._content_token_boundary_cache[cache_key] = (prefix_token_count, suffix_token_count)
        return prefix_token_count, suffix_token_count

    def _build_content_token_mask_from_boundaries(
        self,
        *,
        valid_token_count: int,
        max_sequence_length: int,
        device: torch.device,
        prompt_item: str,
    ) -> torch.Tensor:
        """根据固定模板首尾 token 数构造 user content mask。

        Args:
            valid_token_count: 当前 prompt 的非 padding token 数，也就是 T_i。
            max_sequence_length: tokenizer 最大长度，用于读取边界 cache。
            device: mask 所在设备。
            prompt_item: 仅用于错误信息。

        Returns:
            content_mask: shape [T_i]，True 表示该 token 会进入 adapter。
        """

        prefix_count, suffix_count = self._content_token_boundary_counts(max_sequence_length)
        content_start = prefix_count
        content_end = int(valid_token_count) - suffix_count
        if content_start >= content_end:
            raise ValueError(
                "user content token mask 为空，可能 prompt 为空、被严重截断，或 chat template 边界不匹配，"
                f"valid_token_count={valid_token_count}, prefix={prefix_count}, suffix={suffix_count}, "
                f"prompt={prompt_item!r}"
            )

        content_mask = torch.zeros(int(valid_token_count), dtype=torch.bool, device=device)
        content_mask[content_start:content_end] = True
        return content_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: list[torch.FloatTensor] | None = None,
        negative_prompt_embeds: list[torch.FloatTensor] | None = None,
        max_sequence_length: int = 512,
    ):
        """编码正向/负向 prompt。

        Args:
            prompt: str 或长度 B 的 list[str]。
            prompt_embeds: 可选预编码正向 embeddings，list length B，item [T_i, D]。
            negative_prompt_embeds: 可选预编码负向 embeddings，list length B，item [T_neg_i, D]。

        Returns:
            prompt_embeds:
                list length B，item shape [T_i, D]。
            negative_prompt_embeds:
                CFG 开启时 list length B，item shape [T_neg_i, D]；
                CFG 关闭时为空 list。

        返回 list[Tensor]，每条样本长度可能不同。这一点必须与 Z-Image transformer
        的 list 输入约定保持一致，不能简单 pad 成一个 batch tensor。
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            if len(prompt) != len(negative_prompt):
                raise ValueError("negative_prompt 数量必须与 prompt 一致")
            negative_prompt_embeds = self._encode_prompt(
                prompt=negative_prompt,
                device=device,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = []
        return prompt_embeds, negative_prompt_embeds

    def encode_prompt_with_content_token_masks(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | list[str] | None = None,
        max_sequence_length: int = 512,
    ):
        """编码 prompt，并返回 user content token mask。

        Returns:
            prompt_embeds:
                list length B，第 i 项 shape [T_i, D]。
            negative_prompt_embeds:
                CFG 开启时 list length B，否则为空 list。
            content_token_masks:
                list length B，第 i 项 shape [T_i]，bool。True 表示该 token
                的字符 offset 与原始 user prompt 内容有重叠；chat template、
                role、generation prompt、special token 和 padding 都为 False。
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds, content_token_masks = self._encode_prompt(
            prompt=prompt,
            device=device,
            max_sequence_length=max_sequence_length,
            return_content_token_masks=True,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            if len(prompt) != len(negative_prompt):
                raise ValueError("negative_prompt 数量必须与 prompt 一致")
            negative_prompt_embeds = self._encode_prompt(
                prompt=negative_prompt,
                device=device,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = []
        return prompt_embeds, negative_prompt_embeds, content_token_masks

    def __call__(
        self,
        prompt: str | list[str] | None = None,
        *,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 9,
        target_step: int | None = None,
        guidance_scale: float = 0.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: str | list[str] | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        seeds: Sequence[int] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: list[torch.FloatTensor] | None = None,
        negative_prompt_embeds: list[torch.FloatTensor] | None = None,
        output_type: str = "latent",
        return_dict: bool = False,
        max_sequence_length: int = 512,
    ):
        """运行到 target_step，返回该步的 latent_x1。

        这个方法没有 @torch.no_grad()，因此当 prompt_embeds 来自 adapter 时，
        返回的 latent 仍然保留到 prompt_embeds/adapter 的梯度路径。

        Args:
            prompt:
                原始文本。训练 adapter 时通常不传它，而是直接传 safe prompt_embeds。
            prompt_embeds:
                可选预编码 embeddings，list length B，item shape [T_i, D]。
                如果来自 adapter，必须保持 requires_grad 路径。
            negative_prompt_embeds:
                CFG 使用的负向 embeddings，list length B，item shape [T_neg_i, D]。
            target_step:
                1-based denoise 步数；None 时等于 num_inference_steps。
            latents:
                可选初始噪声，shape [B, 16, H_lat, W_lat]。

        Returns:
            output_type="latent" 且 return_dict=False:
                latent_x1_t，shape [B, 16, H_lat, W_lat]。
            return_dict=True:
                {"latents": latent_x1_t, "target_step": target_step}
        """

        if output_type != "latent":
            raise ValueError("ZImageTrainPipeline.__call__ 只支持 output_type='latent'")

        target_step = num_inference_steps if target_step is None else int(target_step)

        from .config import ProxyLatentConfig
        from .zimage_proxy import ZImageProxyLatentRunner

        config = ProxyLatentConfig(
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            t_min=target_step,
            t_max=target_step,
            guidance_scale=guidance_scale,
            cfg_normalization=cfg_normalization,
            cfg_truncation=cfg_truncation,
            max_sequence_length=max_sequence_length,
        )
        runner = ZImageProxyLatentRunner(self, config)

        if prompt_embeds is None:
            if prompt is None:
                raise ValueError("必须提供 prompt 或 prompt_embeds")
            prompts = [prompt] if isinstance(prompt, str) else prompt
            prompt_embeds, negative_prompt_embeds = runner.encode_prompts(
                prompts,
                negative_prompt=negative_prompt,
            )
        elif guidance_scale > 0 and negative_prompt_embeds is None:
            # prompt_embeds 由外部传入时，不能重算正向 embedding；这里只补 CFG
            # 需要的负向 embedding。传入 prompt_embeds 可让 encode_prompt 直接复用
            # 正向侧，只编码 negative prompt。
            dummy_prompts = ["" for _ in prompt_embeds]
            _, negative_prompt_embeds = self.encode_prompt(
                prompt=dummy_prompts,
                device=self._execution_device,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                max_sequence_length=max_sequence_length,
            )

        latent_x1 = runner.denoise_to_step(
            prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            target_step=target_step,
            seeds=seeds,
            generator=generator,
            latents=latents,
        )

        if return_dict:
            return {"latents": latent_x1, "target_step": target_step}
        return latent_x1

    def _encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        prompt_embeds: list[torch.FloatTensor] | None = None,
        max_sequence_length: int = 512,
        return_content_token_masks: bool = False,
    ):
        """编码单侧 prompt。

        Shapes:
            tokenizer input_ids: [B, max_sequence_length]
            tokenizer attention_mask: [B, max_sequence_length]
            text_encoder hidden_states[-2]: [B, max_sequence_length, D]
            return item i: [T_i, D]，T_i = attention_mask[i].sum()
        """

        device = device or self._execution_device

        if prompt_embeds is not None and return_content_token_masks:
            raise ValueError("传入 prompt_embeds 时无法反推出 user content token mask")
        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        # Z-Image 使用 chat template。这里与现有 pipeline 保持一致，否则 text
        # embedding 分布会和推理/评估不一致。
        formatted_prompts = []
        for prompt_item in prompt:
            formatted_prompts.append(self._format_user_prompt(prompt_item))

        tokenizer_kwargs = {
            "padding": "max_length",
            "max_length": max_sequence_length,
            "truncation": True,
            "return_tensors": "pt",
        }
        text_inputs = self.tokenizer(formatted_prompts, **tokenizer_kwargs)

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        # prompt_hidden: [B, max_sequence_length, D]
        prompt_hidden = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []
        content_token_masks = []
        for index in range(len(prompt_hidden)):
            # prompt_hidden[index][prompt_masks[index]]: [T_i, D]
            embeddings_list.append(prompt_hidden[index][prompt_masks[index]])
            if return_content_token_masks:
                valid_token_count = int(prompt_masks[index].sum().item())
                content_mask = self._build_content_token_mask_from_boundaries(
                    valid_token_count=valid_token_count,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    prompt_item=prompt[index],
                )
                content_token_masks.append(content_mask)
        if return_content_token_masks:
            return embeddings_list, content_token_masks
        return embeddings_list

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """生成或校验初始 latent。

        Args:
            batch_size: B。
            num_channels_latents: Z-Image latent 通道数，当前通常是 16。
            height/width: 目标图像尺寸，会折算为 latent 高宽。
            latents: 可选预生成 latent，shape 必须等于 [B, C, H_lat, W_lat]。

        Returns:
            latents: shape [B, C, H_lat, W_lat]，其中 C=num_channels_latents。

        形状计算与现有 Z-Image pipeline 一致：输入图片尺寸会先按 VAE scale
        折算到 latent 空间，并保证 latent 高宽为偶数。
        """

        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        # shape: [B, C, H_lat, W_lat]
        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        return latents
