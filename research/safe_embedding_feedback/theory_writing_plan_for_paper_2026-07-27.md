# 理论部分论文写法建议

日期：2026-07-27

## 总体原则

理论部分不要写成“我们证明了 diffusion 安全 unlearning 的全局最优”。这个说法过强，容易被审稿人攻击，因为完整 T2I diffusion / flow-matching 生成器是非凸、多阶段、随机、critic 也有误差。

更稳妥、也更符合当前实验的方法表述是：

> We study safe prompt adaptation as a minimum-distortion projection problem under a frozen text-to-image generator and a frozen image-domain latent critic. Under a local linear approximation of the critic-defined risk, the optimal prompt-embedding intervention admits a closed-form minimum-norm solution. Our adapter amortizes this per-prompt intervention into a single forward pass.

中文含义：

> 我们把安全 prompt 修正建模为：在冻结 T2I 生成器和冻结图像域 latent critic 下，寻找一个最小扰动的 prompt embedding 投影。在 critic 定义的局部风险函数满足线性近似时，最优 embedding 干预有闭式最小范数解；我们的 adapter 则把这种 per-prompt 最优干预摊销成一次前向映射。

这套表述足够“理论化”，但没有夸大到全局最优。

## 每个命题的用途

### 命题 A：Text-Proxy Mismatch

**用途**：写动机和相关工作批判。

这条命题回答：为什么只在文本 embedding 空间里用危险词定义 unsafe subspace 不够？

可以写在 Introduction / Motivation：

```text
Existing prompt-embedding sanitization methods often define unsafe directions from unsafe words or unsafe prompts in the text embedding space. However, text-defined unsafe directions are only proxies for the unsafe region induced by a text-to-image generator. Harmful images can be triggered by implicit visual compositions, multi-token interactions, or generation-time binding between text and visual latents. Therefore, the text-defined unsafe subspace may not coincide with the preimage of the true image-domain unsafe region.
```

配套公式：

```text
S_text != G^{-1}(S_img)
```

或者更明确：

```text
U_text = span({unsafe text embeddings})
U_img = {E : C(G(E)) is unsafe}
U_text != U_img
```

**在论文里的作用**：

- 解释 SAFREE / Embedding Sanitizer 的局限；
- 引出你为什么要用 Z-03 latent critic；
- 解释为什么安全文本也可能生成有害图像；
- 解释为什么 text-only projection 会擦不干净或误伤。

**是否需要证明**：主文不需要严格证明，因为这是建模差异和动机。可以在附录给一个线性子空间 mismatch 的 toy theorem，说明如果真实危险方向不在 text proxy 子空间内，就存在残留风险下界。

### 命题 B：Risk Gradient Pullback

**用途**：写方法合理性。

这条命题回答：Z-03 是图像域/latent 域分类器，为什么它能指导 prompt embedding adapter？

主文公式：

```text
r_t(E, eps) = C(F_t(E, eps))
```

其中：

```text
F_t(E, eps) = x_t - sigma_t v_psi(x_t, t, E)
```

链式法则：

```text
grad_E r_t(E, eps)
= J_{F_t}(E)^T grad_z C(z)
```

**在论文里的作用**：

- 说明你的风险方向不是文本词方向，而是 image-domain risk 通过 diffusion trajectory 拉回到 prompt embedding；
- 支撑 “critic-aligned prompt unlearning”；
- 和 AlignProp / classifier guidance / STG 建立联系。

**是否需要证明**：这是链式法则，主文给公式即可，不需要单独 theorem。

### 命题 C：Minimum-Norm Safety Projection

**用途**：这是最重要的主 theorem。

这条命题回答：在特定条件下，为什么你的 residual 形式是最优的？

主文推荐写成 Theorem 1。

问题定义：

```text
min_delta 1/2 ||delta||_M^2
s.t. r(E + delta) <= -m
```

局部线性化：

```text
r(E + delta) ~= r(E) + g^T delta
```

闭式解：

```text
delta*
= - (r(E)+m) / (g^T M^{-1} g) * M^{-1}g
```

**在论文里的作用**：

- 给出“最小扰动修正”的理论保证；
- 支撑 residual adapter，而不是直接生成新 embedding；
- 支撑 hinge margin loss：进入 margin 后不应继续过度推远；
- 解释为什么 embedding regularization 是必要项。

**主文写法**：

```text
Theorem 1 (Local minimum-distortion safety projection).
Assume that the critic-defined risk r(E) is locally linear around a prompt embedding E, i.e., r(E+delta)=r(E)+g^T delta, and semantic distortion is measured by a positive-definite quadratic form ||delta||_M^2. If r(E)+m>0, the minimum-distortion intervention that moves E across the safety margin r(E+delta)<=-m is

delta* = - (r(E)+m)/(g^T M^{-1}g) M^{-1}g.
```

证明放附录，用 KKT 条件即可。

### 命题 D：Amortized Projection Error

**用途**：解释为什么训练 adapter，而不是每个 prompt 在线优化。

这条命题回答：adapter 和 per-prompt optimal update 的关系是什么？

定义 per-prompt 最优修正：

```text
delta*(E) = argmin_delta L_risk(F_t(E+delta)) + lambda ||delta||^2
```

adapter 学：

```text
Delta_theta(E) ~= delta*(E)
```

如果：

```text
||Delta_theta(E) - delta*(E)|| <= eps_A
```

且风险函数 Lipschitz：

```text
|r(E_1)-r(E_2)| <= L_r ||E_1-E_2||
```

则：

```text
r(E+Delta_theta(E)) <= r(E+delta*(E)) + L_r eps_A
```

**在论文里的作用**：

- 说明 adapter 是 per-prompt optimization 的 amortized approximation；
- 给出一个简单误差界；
- 指导实验：可以用 STG / per-prompt gradient search 当 teacher，测 `eps_A`。

**是否进主文**：建议主文只写一句和一个公式，完整 lemma 放附录。

### 命题 E：Proxy / Critic Error Bound

**用途**：解释 target-step `latent_x1` 和 Z-03 critic 的局限。

这条命题回答：我们训练时只看 target-step proxy latent，怎么和最终图像风险关联？

定义：

```text
r_t(E): target-step latent_x1 risk
r_T(E): final generation risk
r_true(E): true oracle risk
```

如果：

```text
|r_T(E)-r_t(E)| <= eps_proxy
|r_true(E)-r_T(E)| <= eps_cls
```

则：

```text
r_true(E) <= r_t(E) + eps_proxy + eps_cls
```

**在论文里的作用**：

- 主动承认 proxy latent 不是最终图像；
- 给出“只要 margin 足够大就安全”的条件；
- 支撑 target step ablation 和 multi-step training。

**是否进主文**：建议主文写简化版，附录写完整误差分解。

### 命题 F：Collateral Damage Lower Bound

**用途**：解释负面现象和局限，不是主贡献。

这条命题回答：为什么擦除目标 IP 可能影响相关概念和风格？

局部线性化：

```text
r(E+delta) ~= r(E) + g_r^T delta
q(E+delta) ~= q(E) + g_q^T delta
```

风险下降至少 `rho` 的最小范数更新：

```text
delta* = -rho g_r / ||g_r||^2
```

对保留概念影响：

```text
|Delta q|
= rho ||g_q|| |cos(g_q, g_r)| / ||g_r||
```

**在论文里的作用**：

- 解释海绵宝宝擦除影响派大星/章鱼哥；
- 解释 IP 擦除导致卡通风格漂移；
- 引出 retain loss、相关概念负样本、IP-conditioned adapter。

**是否进主文**：建议放 Discussion 或 Appendix。主文最多一句：collateral changes are unavoidable when risk and retain directions are entangled。

### 命题 G：SGD Convergence

**用途**：补充理论完整性，不是主要卖点。

非凸神经 adapter 一般只能证明：

```text
min_{0<=k<K} E[||grad J(theta_k)||^2] <= O(1/sqrt(K))
```

**在论文里的作用**：

- 回答“训练能不能收敛”的问题；
- 但这是标准非凸优化结论，不新颖；
- 不建议作为主贡献。

**是否进主文**：不建议。放附录或不写。

## 推荐论文结构

### Introduction

核心叙事：

1. Modern DiT / flow-matching T2I models can generate unsafe content even without explicit unsafe words.
2. Existing text-space sanitization defines unsafe directions using toxic tokens or unsafe prompt corpora.
3. Such text-defined directions are proxies and may mismatch the diffusion-induced unsafe region.
4. We define unsafe knowledge through an image-domain latent critic and learn a prompt adapter that removes the critic-detectable unsafe component with minimal embedding distortion.

可以用这句作为贡献点：

```text
We formulate safe prompt adaptation as image-domain critic-aligned concept erasure in the pullback space of a frozen diffusion generator.
```

### Related Work

分四组：

1. Text-space prompt embedding sanitization：SAFREE、Embedding Sanitizer、CGCE。
2. Diffusion / flow concept erasure：ESD、UCE、EraseAnything、Z-Erase、DVE、MACE。
3. Inference-time DiT safety steering：SafeRoPE、SafeDIG、CAT、UVR。
4. Representation concept erasure theory：INLP、LEACE、Information Bottleneck、disentanglement impossibility。

### Method

方法章节建议分成三小节。

#### 1. Critic-defined Unsafe Preimage

定义：

```text
z_{1,t} = F_t(E, eps)
r_t(E, eps) = C(z_{1,t})
```

然后说明 unsafe region 是：

```text
U_img = {E : r_t(E, eps) > 0}
```

#### 2. Safe Prompt Embedding Adapter

定义：

```text
A_theta(E,c)=E+Delta_theta(E,c)
```

训练目标：

```text
min_theta E[
  L_risk(C(F_t(A_theta(E,c), eps)), y_safe(c))
  + lambda ||A_theta(E,c)-E||^2
  + eta L_retain
]
```

#### 3. Optimization Details

写 target-step sampling、random seed、Z-Image frozen、Z-03 frozen、只训练 adapter。

### Theory

主文 Theory 建议只写 1 页左右。

#### 1. Text-Proxy Mismatch

写成一段动机分析，不一定 theorem。

核心公式：

```text
U_text != U_img
```

#### 2. Local Minimum-Distortion Projection

写 Theorem 1。

只放结论和解释，证明放附录。

#### 3. Error Decomposition

写一个 Proposition 或 Remark：

```text
r_true(A_theta(E))
<=
r_t(E+delta*(E))
+ eps_approx
+ eps_opt
+ eps_proxy
+ eps_cls
+ eps_lin
```

解释每项误差怎么对应实验。

### Experiments

理论对应的实验建议：

1. Safety removal：证明主效果；
2. Benign preservation：验证最小扰动；
3. Related concept preservation：验证 collateral damage；
4. Target step ablation：估计 `eps_proxy`；
5. Adapter capacity / per-prompt teacher comparison：估计 `eps_approx`；
6. Text proxy vs image-domain critic direction cosine：验证 `U_text != U_img`。

### Discussion

放理论边界：

- 不是全局 unlearning；
- 依赖 Z-03 critic 的准确性；
- target-step proxy 和最终图像存在 gap；
- 相关概念纠缠会带来不可避免的 tradeoff。

## 主文可直接使用的理论段落

### Motivation Paragraph

```text
Most prompt-level safeguards define unsafe directions using explicit toxic tokens or unsafe prompt corpora in the text embedding space. However, the unsafe behavior of a text-to-image generator is induced after cross-modal interaction between the text condition and the visual denoising trajectory. Therefore, a text-defined unsafe subspace is only a proxy for the preimage of the image-domain unsafe region under the generator. This mismatch can leave residual unsafe generations when harmful visual concepts are triggered implicitly, and can also remove benign semantics when the text-defined direction entangles safety with style, identity, or context.
```

### Method Paragraph

```text
We instead define unsafe knowledge through a frozen latent safety critic applied to the clean latent estimate of the diffusion trajectory. Given prompt embedding E, the frozen generator induces a target-step latent F_t(E, eps), and the critic defines a risk score r_t(E, eps). The gradient of this risk with respect to E is the pullback of the image-domain risk gradient through the frozen denoising map. Our adapter learns a residual intervention A_theta(E)=E+Delta_theta(E) that reduces this critic-defined risk while minimizing embedding distortion.
```

### Theory Paragraph

```text
Under a local linear approximation of the critic-defined risk, safe prompt adaptation reduces to a constrained minimum-distortion projection problem. Specifically, for a risk margin m and a positive-definite distortion metric M, we seek the smallest intervention delta such that r(E+delta)<=-m. The resulting convex quadratic program admits a closed-form solution, showing that the optimal local intervention follows the negative pullback risk gradient and stops exactly at the safety margin. This provides a theoretical justification for residual prompt adaptation and margin-based risk losses.
```

## 主 Theorem 推荐写法

```text
Theorem 1 (Local minimum-distortion safety projection).
Let r be a critic-defined risk score over prompt embeddings. Assume that around a prompt embedding E, r is locally linear:

    r(E+delta) = r(E) + g^T delta,

where g = grad_E r(E). Let M be a positive-definite matrix defining the distortion ||delta||_M^2 = delta^T M delta. If r(E)+m>0, then the solution of

    min_delta 1/2 delta^T M delta
    s.t. r(E+delta) <= -m

is

    delta* = - (r(E)+m)/(g^T M^{-1}g) M^{-1}g.

Thus, among all interventions that move E across the safety margin, delta* has the minimum distortion under the metric M.
```

证明：

```text
Proof sketch.
The constraint becomes g^T delta <= -r(E)-m under local linearity. Since M is positive definite, the objective is strictly convex and the feasible set is a half-space. The KKT conditions are necessary and sufficient. Setting the gradient of the Lagrangian to zero gives delta = -lambda M^{-1}g. Enforcing the active margin constraint yields lambda=(r(E)+m)/(g^T M^{-1}g), which gives the stated solution.
```

## 附录组织

建议附录这样放：

```text
Appendix A. Proof of Theorem 1
Appendix B. Multi-risk Extension
Appendix C. Error Decomposition
Appendix D. Collateral Damage Bound
Appendix E. Convergence of Per-Prompt Residual Optimization
Appendix F. Empirical Estimation of Proxy and Amortization Errors
```

不要把所有命题塞进主文。主文的理论部分只需要证明“我们的方法在局部条件下是最小扰动安全投影”。其他命题服务于解释和边界。

## 最推荐保留的理论贡献表述

如果论文只能放一句理论贡献，建议写：

```text
We show that, under a local linear approximation of an image-domain latent critic, safe prompt adaptation admits a closed-form minimum-distortion projection in prompt embedding space, and our adapter amortizes this projection across prompts.
```

中文：

```text
我们证明，在图像域 latent critic 的局部线性近似下，安全 prompt 修正可以写成 prompt embedding 空间中的闭式最小扰动投影，而我们的 adapter 将这种 per-prompt 投影摊销到不同 prompt 上。
```
