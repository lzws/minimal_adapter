# Minimal Adapter Runtime

这个目录是可迁移的最小 prompt-embedding adapter 代码包。它只保留：

- Z-Image latent proxy denoise；
- latent detector backbone cosine supervision；
- 5-IP prototype bank；
- 所有当前 adapter 结构的训练；
- 从 checkpoint 自动恢复结构并生成 base/adapter 对比结果。

不包含 Z-Image 权重、latent backbone checkpoint 和 reference bank。

## 支持的 Adapter

`train_adapter.py --adapter_type` 支持：

- `mlp`
- `risk_query_attention_gate`
- `risk_query_film_gate`
- `zimage_adaln_text_condition`
- `zimage_adaln_classifier_condition`
- `bottleneck_self_attention`

普通 MLP 还支持：

- `--gate_type none`
- `--gate_type global`
- `--gate_type token`

## 迁移目录

建议在新服务器准备如下结构：

```text
project_root/
├── minimal_adapter/
├── Z-Image-Turbo/
├── backbone_iter100000.pth
└── ip5_zimage_backbone_s4.pt
```

`minimal_adapter/latent_detector_model.py` 和 `minimal_adapter/backbone_config.json`
已经包含在代码包中，训练和测试 CSV 也已经放在 `minimal_adapter/data/`。

数据文件：

```text
minimal_adapter/data/
├── train/
│   ├── ip_5_z03_filtered_23760.csv
│   ├── benign_all.csv
│   └── single_ip/
│       ├── ip_doraemon_z03_filtered_5000.csv
│       ├── ip_elsa_z03_filtered_4230.csv
│       ├── ip_minion_z03_filtered_5000.csv
│       ├── ip_snow_white_z03_filtered_4530.csv
│       └── ip_spongebob_z03_filtered_5000.csv
├── test/
│   ├── ip_5.csv
│   ├── benign_200.csv
│   ├── benign_200_train_disjoint.csv
│   └── ip_by_category/
└── optional_related_benign/
```

正式测试建议使用 `benign_200_train_disjoint.csv`，避免和训练 benign 集重叠。

运行 Python 前确保当前环境名为 `loraretrieval`。本代码不自动安装依赖；需要的已有依赖包括
`torch`、`torchvision`、`diffusers`、`transformers`、`accelerate`、`pandas` 和 `Pillow`。

## 训练

先做 1-step smoke：

```bash
CUDA_VISIBLE_DEVICES=0 \
conda run -n loraretrieval python -u minimal_adapter/train_adapter.py \
  --model_path Z-Image-Turbo \
  --unsafe_csv minimal_adapter/data/train/ip_5_z03_filtered_23760.csv \
  --benign_csv minimal_adapter/data/train/benign_all.csv \
  --latent_reference_features ip5_zimage_backbone_s4.pt \
  --latent_reference_multi_ip \
  --latent_backbone_checkpoint backbone_iter100000.pth \
  --latent_backbone_model_file minimal_adapter/latent_detector_model.py \
  --output_dir outputs/minimal_smoke_mlp \
  --adapter_type mlp \
  --adapter_depth 1 \
  --gate_type none \
  --t_min 4 --t_max 4 \
  --max_unsafe_samples 20 \
  --max_benign_samples 20 \
  --max_train_steps 1 \
  --save_every 1 \
  --log_every 1
```

正式训练只需要把 `--max_train_steps` 改成目标步数，并去掉两个 `--max_*_samples` 参数。

### MLP

```bash
CUDA_VISIBLE_DEVICES=0 \
conda run -n loraretrieval python -u minimal_adapter/train_adapter.py \
  --model_path Z-Image-Turbo \
  --unsafe_csv minimal_adapter/data/train/ip_5_z03_filtered_23760.csv \
  --benign_csv minimal_adapter/data/train/benign_all.csv \
  --latent_reference_features ip5_zimage_backbone_s4.pt \
  --latent_reference_multi_ip \
  --latent_backbone_checkpoint backbone_iter100000.pth \
  --latent_backbone_model_file minimal_adapter/latent_detector_model.py \
  --output_dir outputs/minimal_mlp \
  --adapter_type mlp \
  --adapter_depth 1 \
  --gate_type none \
  --residual_scale 0.5 \
  --t_min 4 --t_max 4 \
  --max_train_steps 20000
```

### MLP Token Gate

把上面的 adapter 参数替换为：

```bash
--adapter_type mlp --adapter_depth 2 --gate_type token --gate_init 0.5
```

也可以直接用更短的别名：

```bash
--token_gate
```

### Risk Query Attention Gate

```bash
--adapter_type risk_query_attention_gate \
--adapter_depth 2 \
--bottleneck_dim 512 \
--adapter_attention_dim 256 \
--gate_init 0.2
```

### Risk Query FiLM Gate

```bash
--adapter_type risk_query_film_gate \
--adapter_depth 2 \
--bottleneck_dim 512 \
--adapter_attention_dim 256 \
--gate_init 0.5
```

### Z-Image AdaLN Text Condition

```bash
--adapter_type zimage_adaln_text_condition \
--adapter_depth 1 \
--adapter_attention_dim 256 \
--adapter_attention_heads 4
```

### Z-Image AdaLN Classifier Condition

```bash
--adapter_type zimage_adaln_classifier_condition \
--adapter_depth 1 \
--adapter_attention_dim 256 \
--adapter_attention_heads 4
```

### Bottleneck Self-Attention

```bash
--adapter_type bottleneck_self_attention \
--adapter_depth 1 \
--adapter_attention_dim 256 \
--adapter_attention_heads 4 \
--adapter_attention_ffn_multiplier 4
```

所有版本都建议先使用：

```bash
--t_min 4 --t_max 4 --latent_reference_multi_ip
```

这样训练时的 latent step 和 reference prototype 保持一致。unsafe CSV 中的
`sub_category` 会决定使用哪个 IP prototype。

如果 latent-backbone cosine 监督过强，导致正常概念被一起压掉，可以把反馈损失
切到相对 ranking 形式：

```bash
--feedback_loss_type ranking \
--ranking_margin 0.2
```

这个模式会让 adapter 后的 target-prototype similarity 低于同一样本的 base
similarity，而不是直接把绝对 similarity 往低处推。它通常应配合更强 preservation
约束使用，例如：

```bash
--benign_fraction 0.3 \
--w_emb 0.1 \
--w_id 0.5 \
--residual_scale 0.2 \
--restrict_adapter_to_user_content_tokens
```

## 测试

测试脚本从 checkpoint 的 `adapter_config` 自动恢复 adapter 类型，不需要重新指定
`--adapter_type`：

```bash
CUDA_VISIBLE_DEVICES=0 \
conda run -n loraretrieval python -u minimal_adapter/test_adapter.py \
  --adapter_ckpt outputs/minimal_mlp/checkpoints/latest.pt \
  --model_path Z-Image-Turbo \
  --unsafe_csv minimal_adapter/data/test/ip_5.csv \
  --benign_csv minimal_adapter/data/test/benign_200_train_disjoint.csv \
  --output_dir outputs/minimal_mlp_test \
  --prompt_set unsafe \
  --num_prompts 20 \
  --num_inference_steps 9
```

如果已经单独生成过 base 图片，只想生成 adapter 图片，增加：

```bash
--skip_base_generation
```

输出目录包含：

```text
outputs/minimal_mlp_test/
├── base/
├── adapter/
└── compare/
```

按 IP 分组测试：

```bash
--prompt_set unsafe \
--use_all_data \
--group_by_metadata_key original_category \
--group_metadata_values "Snow White,Doraemon,Minions,Elsa,SpongeBob SquarePants" \
--samples_per_group 20
```

## 继续训练

使用和原训练相同的 adapter 参数，并增加：

```bash
--resume_from_checkpoint outputs/minimal_mlp/checkpoints/latest.pt
```

如果只恢复模型参数、不恢复 optimizer，再增加：

```bash
--reset_optimizer_on_resume
```
