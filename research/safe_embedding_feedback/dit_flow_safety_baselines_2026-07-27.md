# DiT / Flow-Matching 安全擦除 Baseline 调研

日期：2026-07-27

## 下载状态

本次已成功下载并校验：

```text
research/safe_embedding_feedback/papers/saferope_2604.01826.pdf
```

`pdfinfo` 校验结果：17 页，题名为 `SafeRoPE: Risk-specific Head-wise Embedding Rotation for Safe Generation in Rectified Flow Transformers`。

本次尝试自动下载 `SAFREE` 时，arXiv PDF 连接速度约几十 KB/s，第一篇 31MB 文件多次超时。因此没有继续批量下载大 PDF。下面给出建议手动补下载清单。

## 建议优先补下载

统一放到：

```text
/mnt/nas2/zhiwen/SafeGuard/baseline/MySLD/research/safe_embedding_feedback/papers/
```

### 必须 baseline

```text
safree_2410.12761.pdf
https://arxiv.org/pdf/2410.12761

eraseanything_2412.20413.pdf
https://arxiv.org/pdf/2412.20413

saferope_2604.01826.pdf
https://arxiv.org/pdf/2604.01826
```

说明：`saferope_2604.01826.pdf` 已下载，无需重复。

### 强相关 DiT / flow-matching 方法

```text
eraseanything_plus_2603.00978.pdf
https://arxiv.org/pdf/2603.00978

dve_2602.01089.pdf
https://arxiv.org/pdf/2602.01089

z_erase_2603.25074.pdf
https://arxiv.org/pdf/2603.25074

safedig_2605.30049.pdf
https://arxiv.org/pdf/2605.30049

flowerase_rl_2605.19739.pdf
https://arxiv.org/pdf/2605.19739

mosaic_2605.25574.pdf
https://arxiv.org/pdf/2605.25574

gem_2606.00140.pdf
https://arxiv.org/pdf/2606.00140
```

### 最新补充 / 理论相关

```text
closed_form_double_projection_2604.10032.pdf
https://arxiv.org/pdf/2604.10032

cgce_2511.05865.pdf
https://arxiv.org/pdf/2511.05865

uni_adavd_2607.14521.pdf
https://arxiv.org/pdf/2607.14521

implicit_negative_embeddings_2511.04834.pdf
https://arxiv.org/pdf/2511.04834

modular_energy_steering_2604.02265.pdf
https://arxiv.org/pdf/2604.02265

casg_2602.20880.pdf
https://arxiv.org/pdf/2602.20880

cat_2603.03163.pdf
https://arxiv.org/pdf/2603.03163

uvr_2606.06875.pdf
https://arxiv.org/pdf/2606.06875
```

## Baseline 优先级

### Priority 1：必须对比

#### SAFREE

- 链接：https://arxiv.org/abs/2410.12761
- 题名：SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image And Video Generation
- 类型：training-free，prompt embedding/text-space toxic subspace + latent re-attention。
- 适合原因：这是你前面一直讨论的 text-space unsafe subspace 路线。它是最直接的对照，可以突出你的方法不是只用危险词定义 proxy 子空间，而是用 Z-03 图像域 latent critic 拉回危险方向。

#### EraseAnything

- 链接：https://arxiv.org/abs/2412.20413
- 题名：EraseAnything: Enabling Concept Erasure in Rectified Flow Transformers
- 类型：flow-based T2I erasure，LoRA 参数更新 + attention map regularizer + self-contrastive preservation。
- 适合原因：明确针对 SD3 / FLUX 这类 flow matching + transformer 架构，是你做 DiT/flow-matching baseline 的核心训练型方法。

#### SafeRoPE

- 链接：https://arxiv.org/abs/2604.01826
- 题名：SafeRoPE: Risk-specific Head-wise Embedding Rotation for Safe Generation in Rectified Flow Transformers
- 类型：training-free，MMDiT head-wise unsafe subspace + RoPE perturbation。
- 适合原因：非常适合作为轻量 inference-time baseline。它干预 transformer attention head 里的 RoPE，而你的方法干预输入 prompt embedding，两者对比点清晰。

#### Z-Erase

- 链接：https://arxiv.org/abs/2603.25074
- 题名：Z-Erase: Enabling Concept Erasure in Single-Stream Diffusion Transformers
- 类型：single-stream diffusion transformer concept erasure。
- 适合原因：摘要明确提到 single-stream DiT，例如 Z-Image。它和你当前 Z-Image setting 最接近，应作为后续强 baseline 或重点相关工作。

### Priority 2：强相关补充

#### Differential Vector Erasure

- 链接：https://arxiv.org/abs/2602.01089
- 题名：Differential Vector Erasure: Unified Training-Free Concept Erasure for Flow Matching Models
- 类型：training-free flow matching erasure。
- 核心点：把概念看成 velocity field 中 target-anchor differential direction，推理时投影掉概念特定速度分量。
- 与你的关系：它在 flow velocity 空间擦除，你在 prompt embedding 空间学习能改变 flow endpoint risk 的 residual。

#### EraseAnything++

- 链接：https://arxiv.org/abs/2603.00978
- 题名：EraseAnything++: Enabling Concept Erasure in Rectified Flow Transformers Leveraging Multi-Object Optimization
- 类型：flow-matching image/video erasure，多目标优化 + implicit gradient surgery。
- 与你的关系：它强调 erasure 和 utility preservation 的冲突，适合作为你后续写多风险/IP 擦除 tradeoff 的相关工作。

#### SafeDIG

- 链接：https://arxiv.org/abs/2605.30049
- 题名：Robust and Generalizable Safety Steering for Text-to-Image Diffusion Transformers
- 类型：DiT safety steering，sparse autoencoder + position-aware feature transfer。
- 与你的关系：它认为 DiT 安全控制不同于 prompt-level 或 output-level 方法，和你的“危险语义在生成轨迹中逐步绑定到视觉 latent”理论动机一致。

#### FlowErase-RL

- 链接：https://arxiv.org/abs/2605.19739
- 题名：FlowErase-RL: Rethinking Concept Erasure as Reward Optimization in Flow Matching Models
- 类型：GRPO-based flow matching concept erasure。
- 与你的关系：你早期考虑过 flow-GRPO。它可以作为 RL 类 baseline 或后续 extension 的相关工作，但实现成本通常高于 adapter。

#### Mosaic

- 链接：https://arxiv.org/abs/2605.25574
- 题名：Mosaic: Compositional Multi-Concept Erasure via Vector Field Blending
- 类型：flow-based multi-concept erasure，vector field blending。
- 与你的关系：适合讨论多 IP / 多概念同时擦除，尤其是同一图中多目标概念的组合场景。

#### GEM

- 链接：https://arxiv.org/abs/2606.00140
- 题名：Geometric Erasure by Contrastive Velocity Matching in Rectified Flows
- 类型：rectified flow erasure，contrastive velocity matching。
- 与你的关系：同样从速度场几何角度解释 erasure，可作为你的“图像域/flow-domain 风险方向比 text proxy 更可靠”的支撑。

## 最新相关论文

下面这些不一定都要跑 baseline，但建议在 related work 里提到。

### Uni-AdaVD

- 链接：https://arxiv.org/abs/2607.14521
- 题名：Universal Concept Erasure for Visual Generation via Orthogonal Value Decomposition
- 时间：2026-07-16
- 特点：inference-time universal erasure，覆盖 U-Net、DiT、autoregressive image generator 和 text-to-video。
- 价值：这是当前列表里最新的一篇，强调跨架构通用性和 orthogonal value decomposition。

### CGCE

- 链接：https://arxiv.org/abs/2511.05865
- 题名：Classifier-Guided Concept Erasure in Generative Models
- 时间：2025-11，2026-07 更新
- 特点：plug-and-play，文本 embedding classifier 检测并 refinement unsafe prompt。
- 价值：和你的“classifier-guided”很接近，但它的 classifier 作用于 text embeddings；你的 critic 作用于 diffusion latent/image-domain response。

### Prompt-Based Safety Guidance Is Ineffective

- 链接：https://arxiv.org/abs/2511.04834
- 题名：Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models
- 特点：指出 prompt-based negative guidance 和 unlearned model 的组合可能无效或退化。
- 价值：可以支持你对 text-space proxy / prompt-only safety 的批判。

### Modular Energy Steering

- 链接：https://arxiv.org/abs/2604.02265
- 题名：Modular Energy Steering for Safe Text-to-Image Generation with Foundation Models
- 特点：使用冻结 foundation models 的 gradient feedback，在 clean latent estimates 上做 energy-based steering；兼容 diffusion 和 flow-matching。
- 价值：与你用冻结 Z-03 critic 训练 adapter 很接近，但它是 inference-time energy steering。

### CASG

- 链接：https://arxiv.org/abs/2602.20880
- 题名：When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance
- 特点：多风险安全方向之间会冲突，动态选择 category-aligned safety direction。
- 价值：适合解释 porn/gore/IP 多风险版本为什么不能简单共享一个安全方向。

### CAT

- 链接：https://arxiv.org/abs/2603.03163
- 题名：Conditioned Activation Transport for T2I Safety Steering
- 特点：用 contrastive safe/unsafe prompt pairs 学 nonlinear transport maps，在 unsafe activation regions 才触发；验证了 Z-Image 和 Infinity。
- 价值：和你当前 Z-Image 设定强相关，尤其是“只在危险区域触发，减少 benign 干扰”的思想。

### UVR

- 链接：https://arxiv.org/abs/2606.06875
- 题名：Unified Safe In-context Image Generation in Multimodal Diffusion Transformers via Restricting Unsafe Information Flows
- 特点：DiT multimodal attention 的 unsafe information flow restriction，覆盖 synthesis 和 editing。
- 价值：适合相关工作里说明 DiT 安全从 prompt-level filtering 转向内部 attention/activation/flow-level 控制。

### Closed-Form Double Projection

- 链接：https://arxiv.org/abs/2604.10032
- 题名：Closed-Form Concept Erasure via Double Projections
- 特点：闭式线性变换，双投影，覆盖 Stable Diffusion 和 FLUX。
- 价值：适合理论相关工作，和 LEACE/INLP 一起支撑“投影式概念擦除”的理论基础。

## 和我们方法的定位差异

### 相比 SAFREE / Embedding Sanitizer

SAFREE 通过 toxic concepts 在 text embedding space 里构造危险子空间，然后 steer prompt embeddings away from this subspace。它的风险方向主要来自文本定义的 unsafe concept set。

我们的方法用：

```text
prompt embedding -> Z-Image denoise -> latent_x1 -> Z-03 risk logits
```

因此危险方向来自 diffusion 生成响应和图像域 latent critic，而不是仅来自危险词。这个差异是核心理论贡献：

```text
text-defined unsafe subspace != diffusion-induced unsafe preimage
```

### 相比 EraseAnything / Z-Erase

EraseAnything / Z-Erase 是模型参数或 LoRA 层面的擦除，优点是擦除更持久；缺点是训练/部署成本更高，且可能影响 base model 的通用能力。

我们方法只训练 prompt embedding adapter，不改 Z-Image backbone。它更像一个 plug-in safety controller：

```text
frozen generator + frozen critic + lightweight adapter
```

这使得我们更容易在不同风险类别上切换，也更容易保留 base model 能力。

### 相比 SafeRoPE / SafeDIG / CAT

这些方法更多干预 transformer 内部 attention head、activation 或 sparse feature。它们更接近模型内部安全控制。

我们的干预点更早：

```text
input prompt embedding
```

但训练信号更晚：

```text
diffusion latent/image-domain risk
```

所以可以强调为：**late-risk-supervised early-intervention**。这也是一个清晰的论文卖点。

### 相比 FlowErase-RL / Flow-GRPO

FlowErase-RL 直接把 flow-matching erasure 写成 reward optimization，并用 GRPO 训练。它理论上更灵活，但训练成本高。

我们的 adapter 是更轻量的 amortized residual policy。可以把它写成 RL/online reward optimization 的前置阶段：

```text
per-prompt reward gradient search -> amortized adapter -> optional RL fine-tuning
```

## 推荐实验对比矩阵

如果后续只跑有限 baseline，建议：

```text
Training-free text-space:
  SAFREE

Training-free DiT/flow internal:
  SafeRoPE
  DVE

Training-based DiT/flow:
  EraseAnything
  Z-Erase

Reward/RL-based:
  FlowErase-RL

Ours:
  Z-03 latent critic trained prompt embedding adapter
```

评估维度：

- unsafe removal rate；
- benign preservation；
- related concept preservation；
- style preservation；
- prompt semantic alignment；
- inference overhead；
- whether backbone weights are modified；
- whether method supports Z-Image/single-stream DiT；
- whether method supports multi-risk conditioning。

## 最建议重点阅读顺序

1. `SAFREE`：理解 text-space unsafe subspace baseline。
2. `SafeRoPE`：理解 MMDiT head-wise intervention。
3. `EraseAnything`：理解 rectified flow transformer erasure 的训练型 baseline。
4. `Z-Erase`：理解 single-stream DiT / Z-Image 的特殊问题。
5. `DVE` 和 `GEM`：理解 velocity-field/flow-domain erasure。
6. `SafeDIG` 和 `CAT`：理解更近的 DiT activation-level safety steering。
