# Z-Image SAFREE Baseline Runner

这个目录是 SAFREE 在 Z-Image 上的正式 baseline runner。

它复现的是官方 SAFREE 中可直接迁移到 Z-Image 的 **text-space concept subspace projection** 主干：

```text
1. 用 unsafe concept prompts 构造危险概念子空间。
2. 对每条 prompt 的 user content token embedding 构造 prompt-specific masked subspace。
3. 将被 SAFREE 选中的 token embedding 投影到危险概念子空间之外。
4. 在指定 denoise step 使用修正后的 prompt embedding 生成图片。
```

官方 repo 中针对 Stable Diffusion UNet upblock 的 Fourier / FreeU-style latent re-attention 不能原样迁移到 Z-Image single-stream transformer，因此这个 runner 不做 UNet upblock patch。它保留了 SAFREE 的 step schedule 接口 `--re_attn_t` 和 self-validation filter `--self_validation_filter`。

## 文件

- `run_zimage_safree_baseline.py`：正式运行入口。
- `safree_text.py`：Z-Image prompt embedding 投影工具。

官方代码来源：

- `baseline/SAFREE`
- `baseline/papers/SAFREE- Training-Free and Adaptive Guard for Safe Text-to-Image And Video Generation.pdf`

## 输出结构

```text
outputs/baselines/zimage_safree/
  images/
    base/              # 加 --run_base 时保存
    safree/            # SAFREE 结果
  compare/             # 加 --save_compare 时保存
  records/
    shards/
    results.jsonl
  run_config.json
  summary.json
```

## 统一 target concept 设置

正式对比实验中，SAFREE 统一使用同一组 target concepts，而不是 porn/gore/IP 分别换不同概念空间。代码里对应 preset 为：

```bash
--unsafe_preset toxic0
```

`toxic0` 包含 porn、gore 和 5 个 IP 概念：

```text
Genitals, Nipples, Exposed breasts, Exposed buttocks,
Sexual intercourse, Oral sex, Penetration, Pornography,
Full frontal nudity, Blood, Gore, Severed body part, Corpse,
Murder, Torture, Elsa, Snow White, Doraemon, Minions,
Spongebob Squarepants
```

## 单卡 smoke test

```bash
CUDA_VISIBLE_DEVICES=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
conda run -n loraretrieval python -u baseline/zimage_safree_runner/run_zimage_safree_baseline.py \
  --input_csv dataset/baseline_eval/porn_level_4_5.csv \
  --output_dir outputs/baselines/zimage_safree/smoke_toxic0_2 \
  --unsafe_preset toxic0 \
  --num_prompts 2 \
  --sample_seed 20260715 \
  --generation_seed 42 \
  --height 512 \
  --width 512 \
  --num_inference_steps 9 \
  --guidance_scale 0.0 \
  --sf_alpha 0.01 \
  --re_attn_t -1,100000 \
  --torch_dtype bfloat16 \
  --device_ids 0 \
  --attention_backend flash \
  --run_base \
  --save_compare \
  --overwrite
```

## 多卡 baseline

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_ALLOC_CONF=expandable_segments:True \
conda run -n loraretrieval python -u baseline/zimage_safree_runner/run_zimage_safree_baseline.py \
  --input_csv dataset/baseline_eval/porn_level_4_5.csv \
  --output_dir outputs/baselines/zimage_safree/porn_toxic0_seed42 \
  --unsafe_preset toxic0 \
  --num_prompts 200 \
  --sample_seed 20260715 \
  --generation_seed 42 \
  --height 512 \
  --width 512 \
  --num_inference_steps 9 \
  --guidance_scale 0.0 \
  --sf_alpha 0.01 \
  --re_attn_t -1,100000 \
  --torch_dtype bfloat16 \
  --device_ids 0,1,2,3 \
  --attention_backend flash \
  --run_base \
  --save_compare
```

## 关键参数

- `--unsafe_preset toxic0/porn/gore/ip`：使用预设 unsafe concept prompts。正式对比默认用 `toxic0`，使 porn/gore/IP/benign 都共享同一个 target concept set。
- `--unsafe_prompts`：用分号分隔的自定义 unsafe concept prompts，设置后覆盖 preset。
- `--sf_alpha`：官方 SAFREE token selection alpha，默认 `0.01`。
- `--safree_scale`：投影强度，默认 `1.0`；如果过度改写可以先试 `0.5`。
- `--re_attn_t start,end`：0-based inclusive denoise step 范围。默认 `-1,100000` 表示所有 step 都使用 SAFREE embedding。
- `--self_validation_filter`：使用官方 SAFREE 的 beta 自适应 step cutoff。
- `--up_t`：self-validation filter 的 step 上界。

## IP 用法

正式对比时，IP 也统一使用：

```bash
--unsafe_preset toxic0
```

这和当前 adapter 使用 Z-03 latent IP classifier feedback 的定义不同。论文对比时需要说明：SAFREE 是 text-space concept subspace baseline，target concepts 由文本列表定义；当前方法的擦除目标由 Z-03 latent/image-domain classifier 定义。
