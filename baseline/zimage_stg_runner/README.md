# Z-Image STG Baseline Runner

这个目录是正式的 Z-Image STG baseline 运行入口。它复用项目已有的：

- `pipelines/ZImageSTGPipeline.py`
- `adapter/clip_safety_model.py`

runner 只负责数据集读取、抽样、多卡分片、图片保存和结果汇总，不修改 Z-Image 或 STG pipeline。

## 输出结构

默认输出到 `outputs/baselines/zimage_stg`：

```text
outputs/baselines/zimage_stg/
  images/
    base/                 # 加 --run_base 时保存
    stg/                  # STG 结果
  compare/                # 加 --save_compare 时保存 base/STG 拼图
  intermediate_x0/        # 加 --save_intermediate_x0 时保存
  records/
    shards/               # 多卡 worker 临时 JSONL
    results.jsonl         # 每条 prompt 的结果记录
  run_config.json         # 本次选中的 prompt 和参数
  summary.json            # 汇总指标
```

## 单卡 porn baseline 示例

```bash
CUDA_VISIBLE_DEVICES=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
conda run -n loraretrieval python -u baseline/zimage_stg_runner/run_zimage_stg_baseline.py \
  --input_csv dataset/trainset-v5-0-sampled-5000/porn_5000.csv \
  --output_dir outputs/baselines/zimage_stg/porn_20_seed42 \
  --unsafe_preset porn \
  --num_prompts 20 \
  --sample_seed 20260715 \
  --generation_seed 42 \
  --height 512 \
  --width 512 \
  --num_inference_steps 9 \
  --guidance_scale 0.0 \
  --lr_upt_prompt 80 \
  --weight_prior 0.01 \
  --update_freq 1 \
  --safety_threshold 0.2 \
  --torch_dtype bfloat16 \
  --device_ids 0 \
  --attention_backend flash \
  --run_base \
  --save_compare
```

## 多卡 gore baseline 示例

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_ALLOC_CONF=expandable_segments:True \
conda run -n loraretrieval python -u baseline/zimage_stg_runner/run_zimage_stg_baseline.py \
  --input_csv dataset/trainset-v5-0-sampled-5000/gore_5000.csv \
  --output_dir outputs/baselines/zimage_stg/gore_200_seed42 \
  --unsafe_preset gore \
  --num_prompts 200 \
  --sample_seed 20260715 \
  --generation_seed 42 \
  --height 512 \
  --width 512 \
  --num_inference_steps 9 \
  --guidance_scale 0.0 \
  --lr_upt_prompt 80 \
  --weight_prior 0.01 \
  --update_freq 1 \
  --safety_threshold 0.2 \
  --torch_dtype bfloat16 \
  --device_ids 0,1,2,3 \
  --attention_backend flash
```

## 指定更新 step

`--update_itrs` 使用 **0-based** denoise step 编号，和 `ZImageSTGPipeline` 内部循环一致：

```bash
--update_itrs 0,1,2
```

如果不设置，默认按 `--update_freq` 更新。例如 `--update_freq 1` 表示每一步都尝试 STG 更新。

## 使用 Z-03 latent classifier 作为 STG 反馈

默认 STG 使用 CLIP image safety feedback。现在也可以改成 Z-03 latent feedback：

```bash
CUDA_VISIBLE_DEVICES=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
conda run -n loraretrieval python -u baseline/zimage_stg_runner/run_zimage_stg_baseline.py \
  --input_csv dataset/trainset-v5-0-sampled-5000/porn_5000.csv \
  --output_dir outputs/baselines/zimage_stg_z03/porn_20_seed42 \
  --safety_feedback z03 \
  --z03_ckpt latent_vit/best_test_avg_f1_model.pth \
  --z03_target_risk porn \
  --z03_loss_type ce \
  --z03_mask_score unsafe_margin \
  --z03_threshold 0.0 \
  --num_prompts 20 \
  --sample_seed 20260715 \
  --generation_seed 42 \
  --height 512 \
  --width 512 \
  --num_inference_steps 9 \
  --guidance_scale 0.0 \
  --lr_upt_prompt 80 \
  --weight_prior 0.01 \
  --update_freq 1 \
  --torch_dtype bfloat16 \
  --device_ids 0 \
  --attention_backend flash \
  --run_base \
  --save_compare
```

Z-03 feedback 不会把 `latent_x1` decode 成图片再分类，而是直接在 STG 的中间 step 使用：

```text
latent_x1 = x_t - sigma_t * velocity
score = Z03(latent_x1)
```

这和当前 adapter 训练里使用 Z-03 的口径一致。

可调参数：

- `--z03_target_risk porn/gore/ip`：选择 Z-03 head。
- `--z03_loss_type ce/softplus_margin/hinge/target_logit/prob`：STG 梯度下降的风险分数。
- `--z03_mask_score unsafe_margin/prob/loss`：判断当前样本是否需要更新 prompt embedding。
- `--z03_threshold`：mask 阈值。`unsafe_margin` 下 0.0 表示 unsafe logit 大于 safe logit 才更新。
- `--z03_ip_loss_mode known_sum/target_class`：IP head 下，压所有 5 个 IP 类，或只压当前样本对应的 IP 类。

注意：Z-03 是 latent classifier，它的反馈发生在 denoise 中间的 proxy latent 上。runner 保存的最终 PIL 图片如果要做正式 Z-03 指标，建议继续使用项目里已有的 Z-03 evaluator 重新生成/记录 target-step latent，再统一统计。

## 自定义 unsafe 文本

默认使用 `--unsafe_preset porn/gore/both`。也可以用分号传入自定义 CLIP unsafe prompts：

```bash
--unsafe_prompts "nudity, explicit sexual content; pornography, erotic image"
```

设置后会覆盖 `--unsafe_preset`。

## 关于 IP baseline

这个 STG runner 当前使用的是 CLIP 文本安全反馈，更适合 porn/gore 这类 safety concept。它可以读取 IP CSV 并生成图片，但这不等价于一个 IP-specific STG baseline，因为默认 CLIP unsafe prompts 并没有 Doraemon/Elsa/SpongeBob 等 IP 类别监督。

如果要把 STG 用作 IP baseline，需要显式提供 IP 相关 unsafe prompts，例如：

```bash
--unsafe_prompts "Doraemon cartoon character; blue robotic cat character"
```

但这仍然是文本 CLIP 反馈，不是你当前方法里的 Z-03 latent IP classifier feedback。因此论文里建议把 STG 主要用于 porn/gore baseline，把 IP 主 baseline 放在 EraseAnything、SafeRoPE、Z-Erase 这类更适合概念/IP 擦除的方法上。
