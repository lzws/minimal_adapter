# Prompt Embedding Adapter 的理论调研与证明方向

日期：2026-07-27

## 目标问题

当前方法可以抽象为：冻结生成模型和安全分类器，只学习一个 prompt embedding 修正器。

给定原始 token embedding：

```text
E in R^{L x d}
```

adapter 输出：

```text
E_safe = A_theta(E, c) = E + Delta_theta(E, c)
```

其中 `c` 是风险类型或 IP 条件。冻结的 Z-Image denoise 到目标 step 后，得到 proxy latent：

```text
z_{1,t}(E_safe, eps) = x_t - sigma_t * v_psi(x_t, t, E_safe)
```

再送入冻结的 Z-03 分类器：

```text
logits = C_phi(z_{1,t})
```

训练目标可以写成：

```text
min_theta E_{p, eps, t, c} [
    L_risk(C_phi(z_{1,t}(A_theta(E_p, c), eps)), y_safe(c))
    + lambda * ||A_theta(E_p, c) - E_p||_M^2
    + eta * L_retain(A_theta(E_p, c), E_p)
]
```

第一版实验里，`L_risk` 主要是 Z-03 head 的 CE 或 hinge risk loss，`||Delta||` 控制语义/风格漂移，`L_retain` 可以是 benign identity loss、CLIP/text similarity 或相关概念保持项。

## 可以主张什么

比较稳妥的理论表述不是“全局最优安全生成”，而是：

> 在冻结生成器、冻结安全 critic、局部线性近似和有限 adapter 表达族下，该方法近似求解一个最小语义扰动的安全投影问题。

更具体：

1. **critic-guided embedding control**：Z-03 loss 的梯度通过 `latent_x1` 和 Z-Image denoise map 反传到 prompt embedding，所以这是 classifier guidance 在条件 embedding 空间里的一个可学习、摊销版本。
2. **minimal-change concept erasure**：如果局部把“危险概念是否可检测”近似为 embedding 空间里的线性函数，最优修正就是沿风险梯度/概念子空间做最小范数投影，和 LEACE/INLP 的可证明线性概念擦除一致。
3. **tradeoff bound**：如果目标概念和保留概念、风格、相关 IP 在表示空间里纠缠，那么任何足够强的风险下降都会带来不可避免的保留损失；这可以解释 IP 擦除时影响相关角色或风格漂移。

不能严谨声称：

- 对真实人类安全偏好的全局最优；
- 对所有攻击 prompt 的完全安全；
- 对非线性生成分布的全局概念解耦。

这些都超出了单个冻结 Z-03 critic 和有限训练数据能证明的范围。

## 与 diffusion / flow 理论的连接

### Diffusion classifier guidance

经典 classifier guidance 在采样过程中修改 score：

```text
score_guided(x_t, t) = score_model(x_t, t) + s * grad_{x_t} log p_phi(y | x_t)
```

我们的做法不是直接改 `x_t`，而是改条件 embedding：

```text
grad_E L_risk
= (partial z_{1,t} / partial E)^T
  (partial C_phi / partial z)^T
  grad_logits L_risk
```

也就是说，Z-03 分类器仍然提供“往安全方向移动”的梯度，但更新变量从 latent/state 变成了 prompt embedding。adapter 进一步把这种 per-prompt 梯度优化摊销成一个一次前向的映射：

```text
E_safe = A_theta(E, c)
```

这能解释为什么训练好以后推理更快：它近似学习了每条 prompt 本来要通过 STG/梯度搜索得到的 embedding 修正。

### Flow matching / rectified flow 视角

Z-Image 这类模型可写成条件流：

```text
dx_t / dt = v_psi(x_t, t, E)
```

训练 flow matching 时，核心是学习从噪声分布到数据分布的 vector field。我们当前不改 `v_psi`，只改条件 `E`，相当于在冻结 vector field 上选择一个更安全的条件轨道：

```text
E -> trajectory tau_E -> endpoint/proxy latent z_{1,t}
```

使用 `latent_x1 = x_t - sigma_t * velocity_t` 作为 Z-03 输入，可以理解为在某个 target step 上估计 endpoint 的安全性，而不是等完整采样结束后再打分。这和 deep reward supervision、STG 在中间采样过程加入 reward 的思想一致。

## 局部最优证明草图

令风险 margin 为：

```text
r(E) = logit_unsafe(E) - logit_safe(E)
```

目标是让 `r(E + delta) <= -m`，同时尽量少改 embedding：

```text
min_delta 1/2 * ||delta||_M^2
s.t. r(E + delta) <= -m
```

在 `E` 附近一阶近似：

```text
r(E + delta) ~= r(E) + g^T delta
g = grad_E r(E)
```

则优化问题变成凸二次规划：

```text
min_delta 1/2 * delta^T M delta
s.t. g^T delta <= -m - r(E)
```

如果当前样本违反安全 margin，即 `r(E) > -m`，KKT 条件给出闭式解：

```text
delta* =
    - (r(E) + m) / (g^T M^{-1} g) * M^{-1} g
```

这个结论的含义：

- 最优方向是风险梯度方向的反方向；
- 修正幅度刚好让样本进入安全 margin；
- 在局部线性和二次范数假设下，它是最小 embedding 扰动；
- hinge loss 比 CE 更接近这个目标，因为进入 margin 后不再继续推离目标概念。

对于多个 IP 或多个风险概念，可以写成：

```text
min_delta 1/2 * delta^T M delta
s.t. G^T delta <= b
```

其中 `G=[g_1,...,g_K]` 是多个风险约束的梯度。最优解是到安全半空间交集的最小距离投影。这可以作为“5-IP 一起擦除”的理论模型。

## 上界：语义和风格漂移

令最终图像分布或 proxy latent map 为：

```text
F_t(E) = z_{1,t}(E, eps)
```

若 `F_t` 和保留度量 `S` 是 Lipschitz 连续的：

```text
||F_t(E + delta) - F_t(E)|| <= L_F ||delta||
|S(F_t(E + delta)) - S(F_t(E))| <= L_S L_F ||delta||
```

因此 embedding residual 的范数直接给出语义/风格漂移上界：

```text
drift <= L_S L_F ||Delta_theta(E, c)||
```

这支持几种工程设计：

- residual adapter 比直接生成完整 embedding 更可控；
- zero-init 让初始模型满足 `Delta=0`，不会一开始破坏原模型；
- hinge margin 能避免风险已经足够低后继续过度修改；
- hard mining 应该优先加“高风险但小扰动可修复”的样本，而不是只追最高风险。

## 下界：概念纠缠导致的不可避免影响

设 `r(E)` 是目标风险，比如“海绵宝宝”，`q(E)` 是希望保留的相关概念或风格，比如“派大星/章鱼哥/卡通风格”。局部线性化：

```text
r(E + delta) ~= r(E) + g_r^T delta
q(E + delta) ~= q(E) + g_q^T delta
```

如果要求风险至少下降 `rho`：

```text
g_r^T delta <= -rho
```

最小范数解会导致保留概念变化：

```text
Delta q ~= -rho * (g_q^T g_r) / ||g_r||^2
```

所以当 `g_q` 和 `g_r` 夹角很小，也就是目标 IP 和相关概念/风格在表示空间里高度相关时：

```text
|Delta q| proportional to |cos(g_q, g_r)|
```

这给出了一个局部下界解释：

- 如果分类器把“海绵宝宝宇宙”的共同视觉特征也当作海绵宝宝证据，擦除海绵宝宝会自然影响派大星、章鱼哥等相关概念；
- 如果目标 IP 的分类证据和“卡通/动漫风格”共享方向，擦除 IP 会导致风格向现实感漂移；
- 这不是 adapter 一定训练坏了，而是风险分类器、数据共现和模型表示共同导致的概念纠缠。

缓解方式必须显式加入保留约束：

```text
min_delta L_risk + lambda ||delta||^2 + eta L_retain_related
```

或者把目标风险梯度投影到保留子空间的正交补：

```text
delta = - alpha * (I - P_retain) g_r
```

其中 `P_retain` 可以由相关概念 prompt、风格 prompt、benign IP-neighbor prompt 的 embedding/gradient 子空间估计。

## 和知识解耦 / unlearning 的关系

传统 diffusion unlearning 多数直接改模型权重，例如 ESD、UCE、ACE、MACE、SalUn。你的方法不改权重，而是做输入侧 unlearning：

```text
model weights fixed
unsafe conditional representation -> safe conditional representation
```

可以把它称为：

- input-side concept erasure；
- prompt-conditional unlearning；
- amortized safe embedding projection；
- classifier-guided representation sanitization。

和权重擦除相比，它的优点是不会永久破坏 base model；缺点是安全性依赖 adapter 是否覆盖攻击分布，以及 critic 是否准确。

## 最相关论文清单

### Prompt embedding / reward 直接相关

- STG: Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models  
  https://arxiv.org/abs/2510.24012  
  与当前方法最接近：采样时用安全函数更新 text embedding，并给出安全约束分布对齐的理论说明。你的 adapter 可以表述为 STG 类 per-prompt 优化的 amortized student。

- PEO: Prompt Embedding Optimization  
  https://arxiv.org/abs/2510.02599  
  说明 prompt embedding 本身可以作为优化变量，目标里包含质量、对齐和 prompt preservation。

- Safe Text-to-Image Generation: Simply Sanitize the Prompt Embedding  
  https://arxiv.org/abs/2411.10329  
  直接主张通过 sanitizing prompt embeddings 提升 T2I 安全性，和你的输入侧 adapter 路线高度相关。

- AlignProp: Aligning Text-to-Image Diffusion Models with Reward Backpropagation  
  https://arxiv.org/abs/2310.03739  
  支持 reward gradient 可以通过 denoising process 反传，而不是必须使用高方差 RL。

- DRTune: Deep Reward Supervisions for Tuning Text-to-Image Diffusion Models  
  https://arxiv.org/abs/2405.00760  
  支持在采样中间步骤加入 reward supervision；这和你用 target-step `latent_x1` 训练非常一致。

- Flow-GRPO: Training Flow Matching Models via Online RL  
  https://arxiv.org/abs/2505.05470  
  支持 flow matching 模型可以在线 RL；但它更适合后期对齐，不一定是 adapter 第一阶段的最稳训练方式。

- SafeDiffusion-R1: Online Reward Steering for Safe Diffusion Post-Training  
  https://arxiv.org/abs/2605.18719  
  近期安全 post-training 工作，使用在线 GRPO 和 embedding-space safety steering，可作为后续 RL 化的参考。

### Diffusion / flow 基础

- DDPM: Denoising Diffusion Probabilistic Models  
  https://arxiv.org/abs/2006.11239

- Score SDE: Score-Based Generative Modeling through Stochastic Differential Equations  
  https://arxiv.org/abs/2011.13456

- Diffusion Models Beat GANs on Image Synthesis  
  https://arxiv.org/abs/2105.05233  
  经典 classifier guidance 来源之一，支撑“分类器梯度可以引导生成”的理论基础。

- Classifier-Free Diffusion Guidance  
  https://arxiv.org/abs/2207.12598  
  说明 guidance 本质上是改变条件分布的采样偏好。

- Flow Matching for Generative Modeling  
  https://arxiv.org/abs/2210.02747

- Rectified Flow: Flow Straight and Fast  
  https://arxiv.org/abs/2209.03003

- EDM: Elucidating the Design Space of Diffusion-Based Generative Models  
  https://arxiv.org/abs/2206.00364

### Concept erasure / diffusion unlearning

- Safe Latent Diffusion  
  https://arxiv.org/abs/2211.05105  
  inference-time safety guidance baseline。

- Erasing Concepts from Diffusion Models  
  https://arxiv.org/abs/2303.07345  
  ESD，使用负向 guidance teacher 进行权重级概念擦除。

- Unified Concept Editing in Diffusion Models  
  https://arxiv.org/abs/2308.14761  
  UCE，闭式编辑 cross-attention projection，适合支撑“闭式最小编辑/多概念编辑”的表述。

- Ablating Concepts in Text-to-Image Diffusion Models  
  https://arxiv.org/abs/2303.13516  
  ACE，强调目标概念和 anchor concept 的分布匹配，并评估相关概念保持。

- Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models  
  https://arxiv.org/abs/2303.17591  
  关注 ID/object/style 的快速 forgetting，也提到 concept correction and disentanglement。

- MACE: Mass Concept Erasure in Diffusion Models  
  https://arxiv.org/abs/2403.06135  
  多概念擦除，尤其适合参考 5-IP 一起擦除时的 generality/specificity tradeoff。

- SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency  
  https://arxiv.org/abs/2310.12508  
  用梯度 saliency 做 unlearning，支持“不是所有参数/维度都应被同等修改”的思想。

### 表示解耦 / 可证明概念擦除

- LEACE: Perfect linear concept erasure in closed form  
  https://arxiv.org/abs/2306.03819  
  这篇最适合拿来写理论证明：在线性可检测概念假设下，存在闭式最小扰动擦除解，且能防止线性分类器检测目标概念。

- INLP: Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection  
  https://aclanthology.org/2020.acl-main.647/  
  通过迭代投影到线性分类器 nullspace 去除属性信息，可作为 token embedding 子空间投影的理论来源。

- beta-VAE / rate-distortion disentanglement  
  https://arxiv.org/abs/1804.03599  
  可用于解释“保留语义”和“压缩/擦除风险信息”的 rate-distortion tradeoff。

- Challenging Common Assumptions in Unsupervised Disentanglement  
  https://arxiv.org/abs/1811.12359  
  重要反例：无监督解耦没有 inductive bias 基本不可能。可以用来说明为什么需要 Z-03 标签、IP condition、benign/related retain set 这些监督约束。

- The Information Bottleneck Method  
  https://arxiv.org/abs/physics/0004057  
  可用于把目标写成“压缩 unsafe 信息，同时保留 prompt/task 相关信息”的信息论形式。

## 对当前实验最有价值的理论路线

### 路线 A：最小扰动安全投影

主公式：

```text
min_delta L_risk(E + delta) + lambda ||delta||^2
```

在局部线性下给出闭式解，证明 residual adapter 学的是最小扰动方向。这条路线最直接、最容易写清楚。

### 路线 B：摊销的 STG / classifier guidance

先定义 per-prompt 最优修正：

```text
delta*(E) = argmin_delta L_risk(F_t(E + delta)) + lambda ||delta||^2
```

adapter 学：

```text
A_theta(E) ~= E + delta*(E)
```

于是 adapter 是 per-prompt optimization 的 amortized approximation。这个角度能解释为什么训练好的 adapter 推理成本低。

### 路线 C：解耦不可能性与保留下界

用梯度相关性说明为什么相关 IP 和风格会被影响：

```text
unavoidable retain drift ~= rho * |cos(g_r, g_q)|
```

这个角度适合解释实验现象，也能自然引出相关概念 retain loss、IP-conditioned adapter、hard mining、token/subspace gating 等改进。

## 推荐下一步阅读顺序

1. 先读 STG、Embedding Sanitizer、LEACE。
2. 再读 classifier guidance、AlignProp、DRTune。
3. 然后读 ESD、UCE、ACE、MACE，整理 unlearning baseline。
4. 最后读 Locatello 和 Information Bottleneck，用来写限制性和 tradeoff。

## 当前 PDF 落盘状态

已有：

- `research/safe_embedding_feedback/papers/stg_2510.24012.pdf`
- `research/safe_embedding_feedback/papers/peo_2510.02599.pdf`
- `research/safe_embedding_feedback/papers/drtune_2405.00760.pdf`
- `research/safe_embedding_feedback/papers/dpok_2305.16381.pdf`
- `research/safe_embedding_feedback/papers/diffexp_2502.14070.pdf`
- `research/safe_embedding_feedback/papers/flow_grpo_2505.05470.pdf`

说明：本次 arXiv PDF 下载出现连接重置，`Flow Matching`、`Rectified Flow`、`Classifier Guidance`、`Classifier-Free Guidance` 的 PDF 文件未可靠落盘，已从本地 papers 目录清理。文档中的论文链接以 arXiv/ACL/OpenReview 主源为准。
