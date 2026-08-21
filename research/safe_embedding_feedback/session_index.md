# 会话索引

这个文件用于给重要 Codex 会话做人工命名，避免只靠 `.codex/history.jsonl`
里的最近消息搜索。这里不修改 Codex 自己的会话存储文件，只在项目内保存可读索引。

## Z-03 Latent Saliency 与 Prompt Embedding Adapter

- 会话名: Z-03 latent saliency + spatial gradient mask adapter
- session_id: `019f640d-e5ed-7970-9104-8df3c623ab58`
- 本地 transcript:
  `/mnt/nas2/zhiwen/.codex/sessions/2026/07/15/rollout-2026-07-15T12-34-20-019f640d-e5ed-7970-9104-8df3c623ab58.jsonl`
- 历史索引:
  `/mnt/nas2/zhiwen/.codex/history.jsonl`
- 工作目录:
  `/mnt/nas2/zhiwen/SafeGuard/baseline/MySLD`
- 关键词:
  `Z-03`, `latent_x1`, `saliency`, `Grad-CAM`, `prompt embedding adapter`,
  `spatial gradient mask`, `IP erasure`, `Doraemon`, `risk_spatial_grad_mask`

### 主要内容

- 基于 Z-03 latent classifier 训练 prompt embedding adapter，将危险或目标 IP prompt embedding 修正到安全方向。
- 明确 Z-03 的输入是 `latent_x1 = latents - sigma_t * velocity`，不是 scheduler step 后的 noisy latent。
- 增加 IP 训练支持，包括 5-IP 数据集、IP condition、target-class loss、related benign preservation 数据集。
- 增加测试和可视化脚本，对比 base model 与 adapter 后生成结果。
- 增加 Z-03 latent saliency 可视化，支持 `gradcam`、`input_gradient`、`grad_x_input`。
- 修正 saliency 可视化解释：heatmap 是 latent 空间相对贡献，不等价于 RGB 像素级“目标区域”。
- 增加 `--heatmap_normalization pair`，让 base/adapter 共用 heatmap scale，减少误读。
- 增加训练时的可选 Z-03 saliency 空间梯度门控：
  `--risk_spatial_grad_mask`、`--risk_spatial_mask_type`、
  `--risk_spatial_mask_topk`、`--risk_spatial_mask_floor`、
  `--risk_spatial_mask_gamma`、`--risk_spatial_mask_score`。

### 相关文件

- `train_prompt_embedding_adapter.py`
- `safe_embedding_adapter/trainer.py`
- `safe_embedding_adapter/config.py`
- `safe_embedding_adapter/losses.py`
- `safe_embedding_adapter/model.py`
- `safe_embedding_adapter/z03.py`
- `experiments/z03_latent_saliency/visualize_z03_latent_saliency.py`
- `test_prompt_embedding_adapter_generation.py`
- `test_prompt_embedding_adapter_runtime_scales.py`

### 查找命令

```bash
grep -n "019f640d-e5ed-7970-9104-8df3c623ab58\\|Z-03\\|saliency\\|risk_spatial_grad_mask" \
  /mnt/nas2/zhiwen/.codex/history.jsonl
```

```bash
grep -n "risk_spatial_grad_mask\\|latent saliency\\|Grad-CAM" \
  research/safe_embedding_feedback/session_index.md
```
