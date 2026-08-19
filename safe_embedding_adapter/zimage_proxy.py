"""Z-Image proxy latent runner。

现有 pipeline 的 __call__ 带 @torch.no_grad()，适合推理但不适合训练 adapter。
本模块复用 pipeline 内部组件，手写一个最小 denoise loop：

    prompt_embeds -> transformer 到目标 step -> latent_x1_t

关键 shape：
    prompt_embeds:       list length B, item [T_i, D]
    initial latents:     [B, 16, H_lat, W_lat]
    transformer input:   list length B, item [16, 1, H_lat, W_lat]
    transformer output:  list length B, item [16, 1, H_lat, W_lat]
    latent_x1_t:         [B, 16, H_lat, W_lat]

Z-03 / latent_vit 使用的不是 scheduler.step 后的下一步 noisy latent，而是当前
step 根据 velocity 一步估计出的 latent_x1：

    latent_x1 = latents - sigma_t * velocity

检测/打标代码里会对 latent_x1 detach；训练 adapter 时不能 detach，否则 Z-03 loss
无法反传到 prompt embeddings 和 adapter。
"""

from __future__ import annotations

from typing import Sequence

import torch

from .config import ProxyLatentConfig


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """复制 Z-Image pipeline 的 shift 计算，避免训练包导入 pipelines/__init__.py。"""

    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return slope * image_seq_len + intercept


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    """复制 diffusers/Z-Image 常用 timestep 获取逻辑。

    这里本可以从 pipeline 文件 import，但 Python 会先执行 pipelines/__init__.py。
    当前仓库的 __init__ 默认导入 SafeRoPE，在 stylus 环境中会触发额外依赖问题。
    因此训练包内部保留这一小段纯函数，减少入口导入副作用。
    """

    import inspect

    if timesteps is not None and sigmas is not None:
        raise ValueError("timesteps 和 sigmas 只能二选一")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"当前 scheduler 不支持自定义 timesteps: {scheduler.__class__}")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_sigmas:
            raise ValueError(f"当前 scheduler 不支持自定义 sigmas: {scheduler.__class__}")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def get_default_z_image_sigmas(num_inference_steps: int) -> list[float]:
    return torch.linspace(1.0, 1 / num_inference_steps, num_inference_steps).tolist()


class ZImageProxyLatentRunner:
    """只负责获取可微 proxy latent。

    pipe 可以是当前项目里的 ZImagePipeline/ZImageSafeRoPEPipeline，只要暴露
    encode_prompt、prepare_latents、scheduler、transformer 这些成员即可。
    """

    def __init__(self, pipe, config: ProxyLatentConfig):
        self.pipe = pipe
        self.config = config

    @property
    def device(self) -> torch.device:
        return torch.device(self.pipe._execution_device)

    def encode_prompts(
        self,
        prompts: Sequence[str],
        *,
        negative_prompt: str | list[str] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """编码 prompt。

        Args:
            prompts: 长度 B 的原始文本列表。
            negative_prompt: CFG 使用的负向 prompt；guidance_scale=0 时不会使用。

        Returns:
            prompt_embeds:
                list 长度 B，第 i 项 shape [T_i, D]。
            negative_prompt_embeds:
                如果 guidance_scale > 0，list 长度 B，第 i 项 shape [T_neg_i, D]；
                否则为空 list。

        text encoder 是冻结的，E_orig 只作为 adapter 输入常量，因此这里使用
        no_grad 可以显著降低显存占用，不影响 adapter 训练。
        """

        do_cfg = self.config.guidance_scale > 0
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt=list(prompts),
                device=self.device,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=negative_prompt,
                max_sequence_length=self.config.max_sequence_length,
            )
        prompt_embeds = [embed.detach() for embed in prompt_embeds]
        negative_prompt_embeds = [embed.detach() for embed in negative_prompt_embeds]
        return prompt_embeds, negative_prompt_embeds

    def encode_prompts_with_content_token_masks(
        self,
        prompts: Sequence[str],
        *,
        negative_prompt: str | list[str] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """编码 prompt，并返回 user content token mask。

        content_token_masks 第 i 项 shape [T_i]，True 表示该 token 来自原始
        user prompt 内容；chat template 和 special token 为 False。
        """

        if not hasattr(self.pipe, "encode_prompt_with_content_token_masks"):
            raise AttributeError("当前 pipeline 不支持 encode_prompt_with_content_token_masks")

        do_cfg = self.config.guidance_scale > 0
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds, content_token_masks = (
                self.pipe.encode_prompt_with_content_token_masks(
                    prompt=list(prompts),
                    device=self.device,
                    do_classifier_free_guidance=do_cfg,
                    negative_prompt=negative_prompt,
                    max_sequence_length=self.config.max_sequence_length,
                )
            )
        prompt_embeds = [embed.detach() for embed in prompt_embeds]
        negative_prompt_embeds = [embed.detach() for embed in negative_prompt_embeds]
        content_token_masks = [mask.detach().to(device=embed.device) for mask, embed in zip(content_token_masks, prompt_embeds)]
        return prompt_embeds, negative_prompt_embeds, content_token_masks

    def _make_generators(self, seeds: Sequence[int] | None) -> list[torch.Generator] | None:
        if seeds is None:
            return None
        generators: list[torch.Generator] = []
        for seed in seeds:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            generators.append(generator)
        return generators

    def prepare_initial_latents(
        self,
        batch_size: int,
        seeds: Sequence[int] | None,
        *,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """准备初始噪声 latent。

        Args:
            batch_size: B。
            seeds: 长度 B 的随机种子列表；None 时由 torch 默认随机源生成。
            generator: 可选 torch generator；如果传入，则不能同时传 seeds。
            latents: 可选预生成 latent，shape 必须是 [B, 16, H_lat, W_lat]。

        Returns:
            latents: shape [B, 16, H_lat, W_lat]。

        使用每条样本固定 seed，保证同一个 prompt 修正前/后可比较。
        """

        num_channels_latents = self.pipe.transformer.in_channels
        if seeds is not None and generator is not None:
            raise ValueError("seeds 和 generator 只能二选一")
        generator = generator if generator is not None else self._make_generators(seeds)
        return self.pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            self.config.height,
            self.config.width,
            torch.float32,
            self.device,
            generator,
            latents,
        )

    def _prepare_timesteps(self, latents: torch.Tensor):
        """根据 latent 空间尺寸准备 scheduler timesteps。

        Args:
            latents: shape [B, 16, H_lat, W_lat]。

        Returns:
            timesteps: shape [num_inference_steps]。
            num_inference_steps: int。
        """

        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = calculate_shift(
            image_seq_len,
            self.pipe.scheduler.config.get("base_image_seq_len", 256),
            self.pipe.scheduler.config.get("max_image_seq_len", 4096),
            self.pipe.scheduler.config.get("base_shift", 0.5),
            self.pipe.scheduler.config.get("max_shift", 1.15),
        )
        sigmas = get_default_z_image_sigmas(self.config.num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler,
            self.config.num_inference_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )
        self.pipe._num_timesteps = len(timesteps)
        self.pipe.scheduler.set_begin_index(0)
        return timesteps, num_inference_steps

    def denoise_to_step(
        self,
        prompt_embeds: Sequence[torch.Tensor],
        *,
        negative_prompt_embeds: Sequence[torch.Tensor] | None = None,
        target_step: int,
        seeds: Sequence[int] | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """运行到 target_step，返回该步的 latent_x1 估计。

        Args:
            prompt_embeds:
                修正后的 safe embeddings，list 长度 B，第 i 项 shape [T_i, D]。
            negative_prompt_embeds:
                CFG 负向 embeddings。guidance_scale=0 时可为 None/空 list；
                guidance_scale>0 时 list 长度 B，第 i 项 shape [T_neg_i, D]。
            target_step:
                1-based denoise 步数。target_step=2 表示执行 step 0 和 step 1。
            seeds:
                长度 B 的随机种子列表，用于生成初始噪声。
            generator:
                可选 torch generator；主要给 pipeline __call__ 风格接口使用。
            latents:
                可选初始噪声，shape [B, 16, H_lat, W_lat]。

        Returns:
            latent_x1_t:
                shape [B, 16, H_lat, W_lat]，会继续送入 Z-03。

        target_step 是 1-based：target_step=2 表示在第 2 个 denoise step
        处捕获 latent_x1。内部对应 Python step_index=1。

        关键区别：
        - 目标 step 之前：执行 scheduler.step，把 x_t 推进到下一步；
        - 目标 step 本身：只计算 latent_x1 = x_t - sigma_t * velocity 并返回，
          不再返回 scheduler.step 之后的 noisy latent。
        """

        if target_step < 1 or target_step > self.config.num_inference_steps:
            raise ValueError(
                f"target_step 必须在 [1, {self.config.num_inference_steps}]，当前为 {target_step}"
            )

        batch_size = len(prompt_embeds)
        if batch_size <= 0:
            raise ValueError("prompt_embeds 不能为空")

        self.pipe._guidance_scale = self.config.guidance_scale
        self.pipe._joint_attention_kwargs = None
        self.pipe._interrupt = False
        self.pipe._cfg_normalization = self.config.cfg_normalization
        self.pipe._cfg_truncation = self.config.cfg_truncation

        latents = self.prepare_initial_latents(batch_size, seeds, generator=generator, latents=latents)
        timesteps, _ = self._prepare_timesteps(latents)

        do_cfg = self.config.guidance_scale > 0
        if do_cfg and not negative_prompt_embeds:
            raise ValueError("guidance_scale > 0 时必须提供 negative_prompt_embeds")

        if do_cfg and self.config.cfg_truncation is not None and float(self.config.cfg_truncation) <= 1:
            precomputed_t_norms = ((1000 - timesteps.float()) / 1000).tolist()
        else:
            precomputed_t_norms = None

        capture_index = target_step - 1
        latent_x1: torch.Tensor | None = None

        for step_index, timestep_raw in enumerate(timesteps[:target_step]):
            timestep = timestep_raw.expand(latents.shape[0])
            timestep_normalized = (1000 - timestep) / 1000

            current_guidance_scale = self.config.guidance_scale
            if precomputed_t_norms is not None and precomputed_t_norms[step_index] > self.config.cfg_truncation:
                current_guidance_scale = 0.0
            apply_cfg = do_cfg and current_guidance_scale > 0

            if apply_cfg:
                # latent_model_input: [2B, 16, H_lat, W_lat]
                # prompt_model_input: list length 2B，前 B 个是正向 embedding，
                # 后 B 个是负向 embedding。
                latent_model_input = latents.to(self.pipe.transformer.dtype).repeat(2, 1, 1, 1)
                prompt_model_input = list(prompt_embeds) + list(negative_prompt_embeds or [])
                timestep_model_input = timestep_normalized.repeat(2)
                actual_batch_size = batch_size
            else:
                # latent_model_input: [B, 16, H_lat, W_lat]
                # prompt_model_input: list length B，item shape [T_i, D]
                latent_model_input = latents.to(self.pipe.transformer.dtype)
                prompt_model_input = list(prompt_embeds)
                timestep_model_input = timestep_normalized
                actual_batch_size = batch_size

            # Z-Image transformer 接受 list 输入。unsqueeze 后：
            # latent_model_input: [B_or_2B, 16, 1, H_lat, W_lat]
            # latent_model_input_list[j]: [16, 1, H_lat, W_lat]
            latent_model_input = latent_model_input.unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))

            # model_out_list: list length B_or_2B，item shape [16, 1, H_lat, W_lat]
            model_out_list = self.pipe.transformer(
                latent_model_input_list,
                timestep_model_input,
                prompt_model_input,
                return_dict=False,
            )[0]

            if apply_cfg:
                pos_out = model_out_list[:actual_batch_size]
                neg_out = model_out_list[actual_batch_size:]
                model_out_items = []
                for sample_index in range(actual_batch_size):
                    pos = pos_out[sample_index].float()
                    neg = neg_out[sample_index].float()
                    pred = pos + current_guidance_scale * (pos - neg)

                    if self.config.cfg_normalization and float(self.config.cfg_normalization) > 0.0:
                        ori_pos_norm = torch.linalg.vector_norm(pos)
                        new_pos_norm = torch.linalg.vector_norm(pred)
                        max_new_norm = ori_pos_norm * float(self.config.cfg_normalization)
                        if new_pos_norm > max_new_norm:
                            pred = pred * (max_new_norm / new_pos_norm)

                    model_out_items.append(pred)
                model_out = torch.stack(model_out_items, dim=0)
            else:
                model_out = torch.stack([item.float() for item in model_out_list], dim=0)

            # model_out: [B, 16, 1, H_lat, W_lat]。
            # 与 latent_vit_ip_detector.py 保持一致：velocity = -model_out.squeeze(2)。
            # velocity: [B, 16, H_lat, W_lat]。
            velocity = -model_out.squeeze(2).float()

            # Z-03 / latent_vit 的输入空间是当前 step 一步估计出的 x1，
            # 而不是 scheduler.step 之后的下一步 noisy latent。
            # 检测/特征库构建会 detach；adapter 训练这里必须保留梯度。
            sigma_t = self.pipe.scheduler.sigmas[step_index].to(device=latents.device, dtype=latents.dtype)
            latent_x1 = latents - sigma_t * velocity.to(dtype=latents.dtype)
            if step_index == capture_index:
                return latent_x1.float()

            # 只有目标 step 之前才需要推进 denoise 轨迹，得到下一步的 x_t。
            latents = self.pipe.scheduler.step(
                velocity.to(torch.float32),
                timestep_raw,
                latents,
                return_dict=False,
            )[0]
            latents = latents.to(torch.float32)

        if latent_x1 is None:
            raise RuntimeError(f"未能在 target_step={target_step} 捕获 latent_x1")
        return latent_x1.float()
