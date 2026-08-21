# Safe Embedding Feedback 方案设计

## 目标

训练一个模型 `f_theta`，输入危险 prompt 的 token embeddings，输出更安全的 prompt embeddings，使得：

- 生成结果尽量避开 unsafe content；
- 语义尽量保留；
- 仍然能接入现有 Z-Image pipeline；
- 后续可以用 reward 继续在线优化。

## 建议的模型形式

不要只输出一个全局向量，建议输出 token-wise residual：

```text
E_safe = E + g_theta(E, m)
```

其中：

- `E` 是原始 prompt embeddings；
- `m` 是可选元信息，例如 prompt 长度、风险类别、是否包含已知敏感 token；
- `g_theta` 输出和 `E` 同形状的残差；
- 可以加 norm clamp 或 low-rank bottleneck，避免语义漂移。

如果要更保守，可以只改“风险 token”附近的 embedding slice，而不是全句都改。

## Reward 设计

建议至少三项：

1. `R_safe`：安全分数，来自 Z-03 porn/gore/ip logits、CLIP safety model 或外部安全分类器。
2. `R_sem`：语义保持，来自 text-image similarity 或原 prompt / safe prompt 一致性。
3. `R_qual`：图像质量或美学分数。

总 reward 可以写成：

```text
R = w1 * R_safe + w2 * R_sem + w3 * R_qual - w4 * ||E_safe - E||
```

如果你要做 RL，建议再加 KL / anchor 项，防止 policy 过度偏离原 prompt 分布。

## 训练路线

### Stage 1: 监督预热

- 用 STG / PEO / 手工 search 得到 teacher safe embeddings。
- 训练 `g_theta` 去拟合 teacher 输出。
- 这一步的目标是让模型先学会“怎么改 embedding”，而不是直接学 reward。

推荐先生成三类 teacher：

- unsafe prompt 经 STG 更新后的 embedding；
- unsafe prompt 经 PEO / gradient search 更新后的 embedding；
- benign prompt 的 identity target，要求 adapter 尽量不改。

### Stage 2: 可微 reward 微调

- 用 `ZImageIPPipeline` 传入 `prompt_embeds`。
- 在固定 `target_step` 捕获 `latent_x1`。
- 把 Z-03 分类 logits 变成 differentiable critic，避免只用 hard threshold。
- 反传到 `g_theta`。

这一步最贴近你现在的代码结构。

注意：`latent_vit/infer_z03_latent.py` 的 `predict()` 是 `torch.no_grad()` 推理函数，适合评估，不适合训练。训练时应复用 `build_z03_model()` 和 `load_checkpoint()`，直接调用模型 forward，保留 transformer 到 latent、再到 Z-03 分类头的梯度路径。

### Stage 3: 在线 RL / flow-GRPO

- 每个 prompt 采样一组候选 safe embeddings。
- 生成对应图像，计算组内相对优势。
- 用 group baseline + KL 约束更新 `g_theta`。

这一步适合后期提质量，不适合第一版从零启动。

如果只训练 `g_theta`，可以先用简化版 group objective：

```text
A_i = R_i - mean(R_group)
L_rl = - mean(A_i * log p_theta(E_safe_i | E)) + beta * KL(E_safe_i, E)
```

如果 `g_theta` 是确定性 adapter，可以把 stochasticity 放在 residual noise、token mask 或 candidate sampler 上，而不是改 diffusion backbone。

## 和现有代码的对接点

- `pipelines/ZImageIPPipeline.py` 已经支持 `prompt_embeds`，适合直接接 student 输出。
- `pipelines/ZImageSTGPipeline.py` 已经做了采样过程中的 embedding 更新，可以作为 teacher。
- `latent_vit/infer_z03_latent.py` 是 Z-03 latent 分类器的完整用法：输入 `[B,16,224,224]` latent，输出 porn/gore/ip logits。
- `adapter/latent_vit_ip_detector.py` 是特征检索式 IP 封装，不应该和 Z-03 分类器主用法混为一谈。
- `run_zimage_latent_vit_ip_detect.py` 更适合做特征检索式离线评估，不适合直接拿来训练 Z-03 reward。

## 需要补的工程点

1. 增加一个 `SafeEmbeddingAdapter`。
2. 增加一个可微 `SafetyRewardModel`。
3. 保存 teacher safe embeddings 轨迹。
4. 提供训练脚本和评估脚本。
5. 对 unsafe / benign prompt 分开统计 false positive。

## 主要风险与处理

- reward hacking：用多 reward 和 KL 约束。
- semantic collapse：只改局部 token，限制残差范数。
- critic 覆盖不足：把 Z-03 三头分类器和通用 safety model 组合。
- train-test mismatch：训练时和推理时都走同一套 Z-Image 采样配置。

## 推荐最小原型

先做这个版本：

1. 输入 prompt embeddings。
2. 输出 token-wise residual safe embeddings。
3. reward 只用 `Z-03 logits + semantic similarity + norm penalty`。
4. 训练方式先监督，再微量 RL。

这个版本最容易在当前仓库里落地，也最容易判断方向值不值得继续加大投入。

## 验证指标

- unsafe removal rate：危险 prompt 生成结果被 critic 判为安全的比例。
- semantic retention：原 prompt 语义是否保留。
- benign preservation：安全 prompt 是否被错误修改。
- image quality：美学分、清晰度或人工偏好。
- reward hacking check：reward 升高时，人工/额外模型是否也认为更安全。

## 建议的首个里程碑

第一周只做一个 Z-03 子任务：用现有 porn/gore/ip logits 做 reward，训练一个小 adapter 降低对应 unsafe 概率。这个任务边界清楚、模型已有、评估链路已有，最适合验证“dangerous embedding -> safe embedding”是否真的能学到。
