# Z-03 评分驱动的 Prompt Embedding 修正模型方案

日期：2026-07-15

## 目标

训练一个轻量模型 `SafeEmbeddingAdapter`，输入原始 prompt embeddings，输出修正后的 safe prompt embeddings，使 Z-Image 生成过程中的 latent 在 Z-03 分类器上更偏向安全类别，同时尽量保持原 prompt 语义。

当前第一版验证只做 porn 色情概念修正：

- 只使用 Z-03 的 `porn_logits` 作为主要评分函数。
- `gore_logits` 和 `ip_logits` 暂不进入训练 loss，只用于可选 logging。
- latent 使用目标 step 的 `latent_x1` proxy，不跑完整最终图像。
- 每个训练样本随机采样一个 denoise step `t`，跑到该 step 后用 `latent_x1_t = x_t - sigma_t * velocity_t` 送入 Z-03。

核心思路：

```text
prompt -> text_encoder -> E_orig
E_orig -> SafeEmbeddingAdapter -> E_safe
sample step t
E_safe -> Z-Image denoise to target step t -> latent_x1_t
latent_x1_t -> Z-03 classifier -> porn_logits
loss(porn_logits, E_safe, E_orig) -> backprop -> SafeEmbeddingAdapter
```

Z-Image 和 Z-03 都冻结，只训练 `SafeEmbeddingAdapter`。

## 现有模块

### Z-03 latent classifier

主接口是 `latent_vit/infer_z03_latent.py`：

- `build_z03_model()`：构建 ConvNeXt-Base + 16 通道 stem + 三个分类头。
- `load_checkpoint(model, ckpt_path)`：加载训练好的 checkpoint。
- 输入：`[B, 16, H, W]` latent，内部或训练代码需要 resize 到 `[B, 16, 224, 224]`。
- 输出：
  - `porn_logits`: `[B, 2]`，`0=safe`, `1=porn`
  - `gore_logits`: `[B, 2]`，`0=safe`, `1=gore`
  - `ip_logits`: `[B, 6]`，`0=no_ip`, `1-5=ip_lv1..ip_lv5`

训练时不要用 `predict()`，因为它是 `@torch.no_grad()` 推理函数，还包含 `argmax` 和字符串 label。训练时要直接调用模型 forward：

```python
z03 = build_z03_model()
z03 = load_checkpoint(z03, ckpt_path).to(device).eval()
for p in z03.parameters():
    p.requires_grad_(False)

porn_logits, gore_logits, ip_logits = z03(latent_224)
```

### Z-Image pipeline

已有 pipeline 支持传入 `prompt_embeds`，这是训练 adapter 的关键入口。

训练时需要注意：

- 不能用包裹整个生成过程的 `torch.no_grad()`。
- 不能在 latent 送入 Z-03 前 `.detach()`。
- Z-Image 参数可以 `requires_grad_(False)`，但 forward 图必须保留，这样 Z-03 loss 能反传到 `E_safe` 和 adapter。

## 固定数据源

当前 porn-only 验证使用两个现成 CSV：

```text
unsafe porn prompts:
/mnt/nas2/zhiwen/SafeGuard/safree_0615/testset_split/porn/level_5.csv

safe / benign prompts:
/mnt/nas2/zhiwen/SafeGuard/safree_0615/testset_split/benign_all.csv
```

读取方式：

- 使用 `prompt` 列作为训练文本。
- 使用 `id` 列作为样本 id。
- 不依赖 `llm_raw_output`，该列包含多行 JSON，不能用简单按行解析。
- 建议用 `pandas.read_csv()` 读取。

当前统计结果：

```text
level_5.csv:
  rows: 1956
  label_porn_risk_level: all 5
  category: porn

benign_all.csv:
  rows: 2692
  label_porn_risk_level: all 0
  label_gore_risk_level: all 0
  label_safe=True: 2408
  label_safe=False: 284, mostly IP risk 6/7
```

对当前 porn-only 实验，`benign_all.csv` 可以整体作为 porn-negative set，因为它的 `label_porn_risk_level` 全部为 0。注意其中有 284 条不是全维度 safe，而是 IP risk 非 0；因为当前不训练 IP head，它们仍可用于“非色情概念保持”验证。如果后续希望 benign set 是全安全语义，应额外过滤：

```text
label_safe == True
```

## 数据流

### 训练样本

每条样本建议包含：

```json
{
  "id": "sample_id",
  "prompt": "raw prompt text",
  "source_csv": "level_5.csv",
  "risk_heads": ["porn"],
  "is_benign": false,
  "porn_target": 0,
  "seed": 42
}
```

字段含义：

- `prompt`：原始文本。
- `source_csv`：样本来自 porn positive 还是 benign negative。
- `risk_heads`：当前第一版固定为 `["porn"]`；以后扩展到 gore/IP 时再打开其他 head。
- `is_benign`：安全 prompt，主要用于 identity preservation，防止 adapter 过度改写。
- `porn_target`：训练目标类别，当前统一为 `0=safe`。unsafe prompt 和 benign prompt 都希望修正后在 Z-03 porn head 上判为 safe。
- `seed`：固定噪声，便于比较修正前后。

### 前向流程

```text
1. prompt
   -> Z-Image text encoder
   -> E_orig: list[Tensor[T_i, D]]

2. E_orig
   -> SafeEmbeddingAdapter
   -> E_safe = E_orig + delta

3. sample denoise step t

4. E_safe
   -> frozen Z-Image denoise for t steps
   -> latent_x1_t: Tensor[B, 16, H, W]

5. latent_x1_t
   -> bilinear resize to [B, 16, 224, 224]
   -> frozen Z-03 classifier
   -> porn_logits

6. porn_logits + E_orig + E_safe
   -> total loss
   -> backprop to SafeEmbeddingAdapter only
```

### latent_z 的选择

第一版有两个可选路径：

1. Full latent：跑完整 denoise，取最终 VAE latent。优点是最贴近最终输出；缺点是慢、显存压力大。
2. Proxy latent：在固定 denoise step 估计 `x1` 或中间 latent。优点是快；缺点是必须确认它和 Z-03 训练用 latent 分布一致。

一般情况下，如果 Z-03 checkpoint 是用最终 VAE latent 训练的，full latent 的分布匹配会更好；如果 Z-03 是用某个 target step 的 `latent_x1` 训练的，target-step `latent_x1` proxy 更自然。当前 porn-only 验证按 Z-03/latent_vit 的口径使用 `latent_x1`，用实验结果判断这个近似是否足够。

当前 porn-only 验证明确采用目标 step 的 `latent_x1` proxy，并且训练时随机 step：

```text
t ~ P(t),  t in [t_min, t_max]
velocity_t = Transformer(x_t, t, E_safe)
latent_x1_t = x_t - sigma_t * velocity_t
```

其中：

- `t_min` 不建议取 0，因为初始噪声几乎不含语义，Z-03 评分会很不稳定。
- `t_max` 不一定等于完整采样步数，可以先取较小范围控制显存和速度。
- 如果总采样步数是 `T`，第一版可以从 `t in [2, T]` 均匀采样开始。
- 后续如果发现早期 step 的 Z-03 评分噪声太大，可以改成偏后期采样，例如 `t in [ceil(0.4T), T]`。

这里的 `latent_x1_t` 不是 `scheduler.step()` 之后的下一步 noisy latent，而是在目标 step 根据当前 `x_t` 和模型 velocity 一步估计出的结果：

```text
latent_x1_t = x_t - sigma_t * velocity_t
```

检测/打标代码里会对 `latent_x1_t` 做 `detach()`；训练 adapter 时不能 detach，否则 Z-03 loss 无法反传到 prompt embeddings 和 adapter。它不要求 decode 成图像，也不要求跑完后续 denoise。

需要记录每次训练的 `t`，用于后续分析不同 step 的 reward 稳定性。

## 当前 Porn-only Proxy 训练数据流

每次训练迭代：

```text
1. 取一个 porn-risk prompt 或 benign prompt
2. 采样 denoise step t
3. prompt -> text encoder -> E_orig
4. E_orig -> SafeEmbeddingAdapter -> E_safe
5. 固定初始 noise seed，使用 E_safe 跑 t 步 Z-Image denoise
6. 捕获 latent_x1_t = x_t - sigma_t * velocity_t
7. resize latent_x1_t 到 [B,16,224,224]
8. frozen Z-03 forward，取 porn_logits
9. 计算 porn safe loss + embedding/identity regularization
10. 反传更新 SafeEmbeddingAdapter
```

训练时可选地同时用 `E_orig` 跑同一个 `t` 做 baseline logging，但不要把这条 baseline 路径放进训练反传，避免显存翻倍。第一版建议只在 eval 阶段做 before/after 对比。

对应的训练目标可以写成：

```text
min_theta E_{prompt, seed, t} [
    L_porn(
        C_z03(
            z_t(E_orig + g_theta(E_orig), seed)
        )
    )
    + regularization
]
```

其中：

- `g_theta` 是 `SafeEmbeddingAdapter`。
- `z_t(...)` 表示使用修正后的 prompt embedding 在目标 step `t` 得到的 `latent_x1` proxy。
- `C_z03` 是冻结的 Z-03 porn head。
- `t` 是随机变量，因此 adapter 学到的是跨多个 denoise 阶段都能降低 porn score 的修正方向，而不是只适配某一个固定 step。

## 随机 Step 策略

第一版使用均匀采样：

```text
t ~ Uniform({t_min, ..., t_max})
```

推荐起点：

```text
T = 9
t_min = 2
t_max = 9
```

不建议从 `t=0` 开始，因为纯噪声 latent 的 Z-03 输出没有稳定语义含义。

如果训练不稳定，可以改成 curriculum：

```text
阶段 1: t in [ceil(0.7T), T]
阶段 2: t in [ceil(0.4T), T]
阶段 3: t in [2, T]
```

这样先让 adapter 在较接近最终图像的 latent 上学会降低 porn score，再逐步覆盖更早 step。

如果显存或实现复杂度限制较大，每个 batch 可以共享同一个 `t`；如果 pipeline 支持灵活中断，每个样本独立采样 `t` 会更接近目标分布。

## Adapter 设计

Z-Image 的 prompt embeddings 是变长 token embeddings，建议 adapter 输出 token-wise residual：

```text
delta_i = g_theta(E_orig_i, risk_condition)
E_safe_i = E_orig_i + alpha * clamp(delta_i)
```

最小结构：

```text
LayerNorm(D)
Linear(D, r)
GELU
Linear(r, D)
Tanh
```

建议：

- `r` 取 `D/4` 或更小，先做小模型。
- `alpha` 从小值开始，例如 `0.05` 到 `0.2`。
- 对 benign prompts 强化 identity loss。
- 可以给 `risk_heads` 加一个小 embedding，作为 adapter 的条件输入。

不要第一版就训练全量 text encoder 或 Z-Image backbone。

## 损失函数设计

总损失：

```text
L_total =
    w_z03 * L_z03
  + w_emb * L_embedding_reg
  + w_sem * L_semantic
  + w_id  * L_identity
  + w_teacher * L_teacher
```

第一版可以只启用前三项：

```text
L_total = L_z03 + 0.05 * L_embedding_reg + 0.1 * L_identity
```

权重需要根据 Z-03 loss 的量级调。

### 1. Z-03 安全损失

最直接形式是 cross entropy，把三个 head 都推向安全类别：

```python
target_safe_2 = torch.zeros(B, dtype=torch.long, device=device)
target_safe_ip = torch.zeros(B, dtype=torch.long, device=device)

L_porn = F.cross_entropy(porn_logits, target_safe_2)
L_gore = F.cross_entropy(gore_logits, target_safe_2)
L_ip = F.cross_entropy(ip_logits, target_safe_ip)

L_z03 = wp * L_porn + wg * L_gore + wi * L_ip
```

如果有 `risk_heads` 标注，可以按任务只打开相关 head：

```text
L_z03 = I_porn * wp * L_porn + I_gore * wg * L_gore + I_ip * wi * L_ip
```

对未知风险或混合风险样本，可以三个 head 都打开。

当前 porn-only 验证只启用 `L_porn`：

```python
target_safe = torch.zeros(B, dtype=torch.long, device=device)
L_porn_ce = F.cross_entropy(porn_logits, target_safe)
```

总 loss 暂时写成：

```text
L_total =
    w_porn * L_porn_ce
  + w_emb  * L_embedding_reg
  + w_id   * L_identity
  + w_sem  * L_semantic
```

第一版建议：

```text
w_porn = 1.0
w_emb  = 0.05
w_id   = 0.1
w_sem  = 0.0 或 0.05
```

`w_sem` 可以先不开，只依赖 embedding residual 和 benign identity 控制语义漂移。

### 2. Margin 形式的安全损失

CE 简单稳定，但有时会过度推高 safe 类。可以换成 margin loss：

```python
L_porn_margin = F.softplus(porn_logits[:, 1] - porn_logits[:, 0] + margin).mean()
L_gore_margin = F.softplus(gore_logits[:, 1] - gore_logits[:, 0] + margin).mean()

ip_unsafe = torch.logsumexp(ip_logits[:, 1:], dim=1)
ip_safe = ip_logits[:, 0]
L_ip_margin = F.softplus(ip_unsafe - ip_safe + margin).mean()
```

这个形式更像“只要 unsafe 比 safe 低 enough 就行”，通常更不容易把 embedding 推得过远。

当前 porn-only 版本也可以只用 porn margin：

```python
margin = 0.5
L_porn_margin = F.softplus(porn_logits[:, 1] - porn_logits[:, 0] + margin).mean()
```

建议实验顺序：

1. 先用 `L_porn_ce` 跑通训练。
2. 如果发现 embedding 被推得过猛，再换成 `L_porn_margin`。
3. 如果 CE 和 margin 都有效，可以比较二者在 benign prompt 上的误伤率。

### 3. Embedding 残差约束

控制 adapter 改写幅度：

```python
L_embedding_reg = mean_i(||E_safe_i - E_orig_i||_2^2 / T_i)
```

可以加 token smoothness：

```python
L_smooth = mean_i(||delta_i[1:] - delta_i[:-1]||_2^2)
```

第一版可以先不加 smoothness。

### 4. 语义保持损失

最低成本版本是 embedding cosine：

```python
pool_orig = mean_pool(E_orig_i)
pool_safe = mean_pool(E_safe_i)
L_semantic = 1 - cosine(pool_orig, pool_safe)
```

更强版本：

- 用 CLIP / T5 embedding 比较原 prompt 与修正后生成图像的语义一致性。
- 用原 prompt 和修正 prompt 生成图像的 CLIP similarity 差值做约束。

第一版建议只用 embedding cosine + residual norm，避免系统过重。

### 5. Benign identity loss

对安全 prompt，adapter 应尽量不动：

```python
L_identity = I_benign * mean_i(||E_safe_i - E_orig_i||_2^2 / T_i)
```

这个 loss 很重要，否则模型可能学成“所有 prompt 都往空泛安全方向推”。

Porn-only 验证里，benign 样本建议也可以保留一个弱 porn-safe loss：

```text
L_benign = lambda_id * L_identity + lambda_safe * L_porn_ce
```

其中 `lambda_safe` 要小，例如 `0.1`，避免 benign prompt 被过度优化。

### 6. Teacher distillation loss

如果先用 STG / PEO 生成了 teacher safe embeddings：

```python
L_teacher = mean_i(||E_safe_i - E_teacher_i||_2^2 / T_i)
```

建议训练顺序：

1. 先用 `L_teacher + L_identity` 预热 adapter。
2. 再加入 `L_z03` 微调。
3. 最后才考虑 RL / group objective。

## 训练阶段

### Stage 0: 数据准备

- 从 `/mnt/nas2/zhiwen/SafeGuard/safree_0615/testset_split/porn/level_5.csv` 读取 porn-risk prompts。
- 从 `/mnt/nas2/zhiwen/SafeGuard/safree_0615/testset_split/benign_all.csv` 读取 benign prompts。
- 每条样本固定 seed，方便做前后对比；训练时 step `t` 随机，seed 可以固定或按 epoch 派生。
- 可选：缓存 `E_orig`，减少 text encoder 开销。

推荐第一版数据配比：

```text
porn-risk prompts: 1956 available
benign prompts:    2692 available
```

推荐划分：

```text
train: 80%
val:   10%
test:  10%
```

划分时分别对 porn-risk 和 benign 两个集合独立 split，再合并，避免某个 split 里正负比例失衡。

推荐 batch 组成：

```text
50% porn-risk
50% benign
```

如果训练早期 porn loss 下降太慢，可以临时提高 porn-risk 比例到 70%，但最终评估必须看 benign prompt 是否被过度改写。

smoke test：

```text
porn-risk: 100
benign:    100
```

先用 smoke test 确认梯度、loss 和随机 step 训练流程都正确，再跑全量。

### Stage 1: Adapter 监督预热

输入 `E_orig`，输出 `E_safe`。

loss：

```text
L = L_teacher + lambda_id * L_identity + lambda_emb * L_embedding_reg
```

如果没有 teacher，可以跳过 Stage 1，但训练会更不稳定。

### Stage 2: Porn-only latent_x1 proxy 可微评分微调

冻结：

- Z-Image
- Z-03
- text encoder

训练：

- SafeEmbeddingAdapter

loss：

```text
L = L_z03 + lambda_emb * L_embedding_reg + lambda_id * L_identity
```

这一步是第一版核心实验。

当前第一版中：

```text
L_z03 = L_porn_ce 或 L_porn_margin
```

每个样本训练时随机采样一个 step `t`：

```text
t ~ Uniform({t_min, ..., t_max})
```

然后只跑到目标 step `t`，捕获 `latent_x1_t = x_t - sigma_t * velocity_t`，送入 Z-03 的 porn head。

推荐初始配置：

```text
T = 9 或当前 Z-Image Turbo 常用采样步数
t_min = 2
t_max = T
P(t) = uniform
```

如果早期 latent 的 porn logits 波动很大，可以改为：

```text
t_min = ceil(0.4 * T)
t_max = T
```

每个 batch 里可以每个样本独立采样 `t`。如果实现上变长 denoise 不方便，第一版可以每个 batch 共享一个 `t`，降低工程复杂度。

### Stage 3: 可选 group RL

如果 Stage 2 有效果但语义/质量不够，可以对每个 prompt 采样多个 candidate residual：

```text
A_i = R_i - mean(R_group)
L_group = - mean(A_i * log_prob(delta_i)) + beta * KL(E_safe_i, E_orig_i)
```

第一版不建议直接做这一阶段。

## 评估指标

### Z-03 指标

对 unsafe prompts：

- porn unsafe probability before/after
- porn safe argmax rate
- porn margin: `porn_logits[:,0] - porn_logits[:,1]`

对 benign prompts：

- porn safe 保持率
- false positive change
- embedding delta norm

因为训练使用随机 step `t`，评估必须分 step 统计：

```text
for t in eval_steps:
    compare E_orig vs E_safe at same seed and same t
```

建议 `eval_steps = [2, 4, 6, 8, T]`，或者覆盖训练用的所有 step。

### 图像指标

- 人工抽查修正前后图像。
- CLIP text-image similarity。
- aesthetic / quality score，可选。
- 失败样本按 porn/gore/ip 分类统计。

## 实现注意点

1. Z-03 参数冻结，但 forward 不能放在 `torch.no_grad()` 里，否则 loss 无法回传到 adapter。
2. Z-Image 参数冻结，但 denoise forward 图要保留。
3. latent 进入 Z-03 前 resize 用 `F.interpolate`，这是可微的。
4. 不要用 `argmax` 参与训练，只用于 logging。
5. 如果显存不够，先减少 batch size、采样步数、分辨率，或者只对后半段 denoise 做 truncated backprop。
6. 第一版只训练 adapter，避免同时改 text encoder / diffusion backbone 导致问题不可定位。

## 最小可行实验

第一版建议：

- 数据：porn-risk 100-300 条起步，benign 100-300 条。
- 模型：两层 MLP residual adapter。
- latent：目标 step 的 `latent_x1` proxy，每次随机采样 step `t`，只跑到该 step 并计算 `x_t - sigma_t * velocity_t`。
- loss：`porn CE safe loss + residual L2 + benign identity`。
- 训练：1 GPU，小 batch，固定 seed。
- 评估：同一批 prompt 修正前后在多个 `t` 上跑 Z-03 porn head + 保存少量完整生成图像人工检查。

成功标准：

- porn-risk prompts 的 Z-03 porn probability 明显下降；
- benign prompts 的 embedding delta 很小；
- 图像语义没有大面积丢失；
- 不出现“所有 prompt 都变成无意义安全图”的 collapse。
