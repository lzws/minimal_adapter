# Safe Prompt Embedding Adapter 的理论解释与可证明命题

日期：2026-07-27

## 结论先行

当前方法可以从理论上解释为：

> 在冻结 diffusion 生成器和冻结图像域安全 critic 下，学习一个从原始 prompt embedding 到安全 prompt embedding 的最小扰动投影器。这个投影器不是基于危险词定义的文本子空间，而是基于 diffusion 生成响应和 Z-03 latent critic 拉回到 prompt embedding 空间的图像域危险方向。

更短的命名可以是：

- diffusion-pullback concept erasure；
- image-domain critic-aligned prompt unlearning；
- amortized safe embedding projection；
- classifier-guided knowledge disentanglement in prompt embedding space。

后续可以证明三类结果：

1. **局部最优性**：在局部线性风险和二次扰动度量下，最优修正有闭式解，是最小范数的安全投影。
2. **收敛性**：如果直接优化每个 prompt 的 residual，凸化后的目标可收敛到全局最优；如果训练神经 adapter，可以在标准 smoothness 假设下证明 SGD 收敛到一阶驻点。
3. **误差界**：最终风险可以分解为 critic 误差、target-step proxy 误差、局部线性化误差、adapter 摊销误差和优化误差；其中每一项都可以用实验估计。

不建议声称“全局最优 unlearning”或“完全知识删除”。这在非凸 diffusion 生成器、有限 critic、有限数据下很难成立。更稳妥的说法是：**在固定生成器和固定 critic 定义的安全任务上，我们的方法近似求解一个最小扰动安全投影问题**。

## 1. 为什么文本空间投影不够

设 prompt embedding 为：

```text
E in H
```

冻结 diffusion 生成器在目标 step 的 proxy latent 映射为：

```text
F_t(E, eps) = z_{1,t}
```

当前代码里：

```text
z_{1,t} = x_t - sigma_t * velocity_t
```

冻结 Z-03 latent classifier 定义图像域风险分数：

```text
r_t(E, eps) = logit_unsafe(C(F_t(E, eps))) - logit_safe(C(F_t(E, eps)))
```

真实应该规避的是图像域危险集合在 prompt embedding 空间里的拉回：

```text
U_img = {E : r_t(E, eps) > 0}
```

而 SAFREE / Embedding Sanitizer 类方法更接近在文本侧定义一个 proxy 危险子空间：

```text
U_txt = span({unsafe word embeddings, unsafe prompt embeddings, text-only directions})
```

关键问题是：

```text
U_txt != U_img
```

原因是 text-to-image 生成存在跨模态非线性映射。危险图像可能由隐含组合触发，而不一定由显式危险词触发；反过来，某些危险词方向也可能携带风格、构图、角色身份等无关信息。

所以文本 proxy 投影会有两种结构性失败：

1. **残留风险**：如果真实危险方向不在文本 proxy 子空间里，仅投影文本危险词方向无法完全压低图像域风险。
2. **误伤无关语义**：如果文本 proxy 子空间混入了风格、角色、场景等保留因素，投影会同时削弱这些因素。

我们的优势在于风险信号来自：

```text
E -> diffusion denoise -> latent_x1 -> Z-03 classifier
```

因此学习到的是图像域危险响应在 prompt embedding 空间里的拉回方向，而不是仅由危险词定义的文本方向。

## 2. 知识解耦表述

可以把 prompt embedding 的局部变化分解为：

```text
delta = delta_risk + delta_retain + delta_null
```

其中：

- `delta_risk` 是会改变 Z-03 图像域风险分数的方向；
- `delta_retain` 是会改变保留语义、风格、相关角色的方向；
- `delta_null` 对当前生成和 critic 影响较小。

这个分解不是全局线性真分解，而是由当前点 `E`、当前噪声 `eps`、当前 step `t` 决定的局部切空间分解。

风险方向由链式法则给出：

```text
g_r(E) = grad_E r_t(E, eps)
       = J_F(E)^T grad_z r(z)
```

其中：

- `J_F(E)` 是 diffusion proxy latent 对 prompt embedding 的 Jacobian；
- `grad_z r(z)` 是 Z-03 critic 在 latent 空间里的风险梯度。

这就是“图像域危险知识拉回到 embedding 空间”的数学形式。adapter 学习的不是危险词，而是这个拉回后的风险响应。

从信息论角度也可以写成：

```text
min_A I(A(E); Y_risk_img)
max_A I(A(E); Y_retain)
s.t. E[d(A(E), E)] <= tau
```

实际训练中用可微 surrogate：

```text
min_theta E[
    L_risk(C(F_t(A_theta(E)))) + lambda ||A_theta(E)-E||^2 + eta L_retain
]
```

其中 `Y_risk_img` 不是文本标签，而是 diffusion 生成后由图像域/latent critic 定义的风险标签。

## 3. 命题一：局部最小扰动最优性

令 adapter 对单个 prompt 的修正为：

```text
E_safe = E + delta
```

定义风险 margin：

```text
r(E) = logit_unsafe(E) - logit_safe(E)
```

希望修正后满足：

```text
r(E + delta) <= -m
```

同时尽量少改原 embedding：

```text
min_delta 1/2 * ||delta||_M^2
s.t. r(E + delta) <= -m
```

在 `E` 附近一阶展开：

```text
r(E + delta) ~= r(E) + g^T delta
g = grad_E r(E)
```

问题变成凸二次规划：

```text
min_delta 1/2 * delta^T M delta
s.t. g^T delta <= -r(E) - m
```

若当前样本违反 margin，即：

```text
r(E) + m > 0
```

KKT 条件给出闭式最优解：

```text
delta* =
    - (r(E) + m) / (g^T M^{-1} g) * M^{-1} g
```

这个解有三个直接含义：

1. 最优方向是风险梯度的反方向。
2. 最优幅度刚好把样本推到安全 margin 边界。
3. 在局部线性和二次范数条件下，这是所有可行修正里语义扰动最小的修正。

因此，使用 residual adapter：

```text
A_theta(E) = E + Delta_theta(E)
```

比直接生成一个全新 embedding 更容易获得理论解释：它近似学习 `delta*`，即最小扰动安全投影。

### 多风险版本

对 porn、gore、多个 IP 同时擦除时，设有 `K` 个风险约束：

```text
r_k(E + delta) <= -m_k,  k=1,...,K
```

局部线性化后：

```text
g_k^T delta <= -r_k(E) - m_k
```

写成矩阵形式：

```text
G^T delta <= -a
```

其中：

```text
G = [g_1, ..., g_K]
a_k = r_k(E) + m_k
```

若一组 active constraints 同时生效，闭式解为：

```text
delta* = - M^{-1} G_A (G_A^T M^{-1} G_A)^{-1} a_A
```

这说明 5-IP 一起擦除可以被解释为：把 embedding 投影到多个图像域风险半空间的交集。不同 IP 之间互相干扰，本质上来自这些约束梯度不正交。

## 4. 命题二：文本 proxy mismatch 的残留风险下界

设真实图像域风险方向为单位向量：

```text
g_img = grad_E r_img(E) / ||grad_E r_img(E)||
```

文本 proxy 方法只能在文本危险子空间 `S_txt` 内修正：

```text
delta in S_txt,  ||delta|| <= R
```

它能获得的最大一阶风险下降为：

```text
max -g_img^T delta = R ||P_txt g_img||
```

其中 `P_txt` 是到文本 proxy 子空间的正交投影。

若希望风险下降至少为 `rho`，则文本 proxy 方法可行的必要条件是：

```text
rho <= R ||P_txt g_img||
```

如果：

```text
rho > R ||P_txt g_img||
```

则残留风险至少为：

```text
residual >= rho - R ||P_txt g_img||
```

若 `theta` 是真实图像风险方向和文本 proxy 子空间的夹角：

```text
||P_txt g_img|| = cos(theta)
```

则：

```text
residual >= rho - R cos(theta)
```

这给出一个很直接的理论解释：

- 如果文本 proxy 子空间和真实 diffusion 危险方向不一致，纯文本投影必然擦不干净；
- mismatch 越大，`cos(theta)` 越小，残留风险下界越高；
- 你的方法用 Z-03 通过 diffusion 轨迹提供 `g_img`，因此比危险词子空间更接近真实风险方向。

这个命题是最适合用来批判 SAFREE / text-only embedding projection 的理论点。

## 5. 命题三：相关概念和风格误伤下界

设目标风险方向为 `g_r`，需要保留的概念或风格方向为 `g_q`。例如：

- 风险方向：海绵宝宝；
- 保留方向：派大星、章鱼哥、卡通风格、水下背景、黄色配色等。

局部线性化：

```text
r(E + delta) ~= r(E) + g_r^T delta
q(E + delta) ~= q(E) + g_q^T delta
```

如果要求风险下降至少 `rho`：

```text
g_r^T delta <= -rho
```

最小范数风险修正为：

```text
delta* = - rho * g_r / ||g_r||^2
```

它对保留概念的影响为：

```text
Delta q = g_q^T delta*
        = - rho * (g_q^T g_r) / ||g_r||^2
```

因此：

```text
|Delta q| = rho * ||g_q|| * |cos(g_q, g_r)| / ||g_r||
```

如果目标 IP 和相关概念、风格在模型内部表示中高度纠缠，即 `cos(g_q, g_r)` 很大，那么任何足够强的擦除都会不可避免地影响相关概念。

这可以解释当前 5-IP 训练中观察到的现象：

- 擦除海绵宝宝时，派大星、章鱼哥也受到影响；
- 擦除 IP 时，动漫/卡通风格被削弱，生成结果变得更现实；
- 这不是单纯的训练 bug，而是风险方向和保留方向在 diffusion 表示空间里重叠。

缓解方式需要显式引入保留约束：

```text
min_delta L_risk + lambda ||delta||^2 + eta L_retain
```

或做正交投影：

```text
delta = - alpha * (I - P_retain) g_r
```

其中 `P_retain` 由相关概念 prompt、风格 prompt、benign prompts 的梯度或 embedding 子空间估计。

## 6. 命题四：adapter 摊销误差界

先定义每个 prompt 的理想最优修正：

```text
delta*(E) = argmin_delta L_risk(F_t(E + delta)) + lambda ||delta||^2
```

训练 adapter 是学习一个摊销近似：

```text
Delta_theta(E) ~= delta*(E)
```

假设风险函数 `r(E)` 是 `L_r`-Lipschitz：

```text
|r(E_1) - r(E_2)| <= L_r ||E_1 - E_2||
```

若 adapter 的摊销误差满足：

```text
||Delta_theta(E) - delta*(E)|| <= eps_A
```

且理想解满足：

```text
r(E + delta*(E)) <= -m
```

则 adapter 输出满足：

```text
r(E + Delta_theta(E)) <= -m + L_r eps_A
```

也就是说，adapter 的安全 margin 损失至多和摊销误差线性相关。

如果考虑期望形式：

```text
E[||Delta_theta(E) - delta*(E)||] <= eps_A
```

则有：

```text
E[r(E + Delta_theta(E))]
<= E[r(E + delta*(E))] + L_r eps_A
```

这个误差界可以指导实验：

- 用 STG / per-prompt gradient search 得到 `delta*(E)` 的近似；
- 测量 adapter 输出和 per-prompt optimum 的距离；
- 该距离越小，理论上安全 margin 越接近 per-prompt optimum。

## 7. 命题五：target-step proxy 和最终图像风险的误差界

训练时用 target-step proxy：

```text
r_t(E) = C(F_t(E))
```

但真实关心的是完整生成后的风险：

```text
r_T(E) = C(F_T(E))
```

如果两者误差有界：

```text
|r_T(E) - r_t(E)| <= eps_proxy
```

并且 Z-03 critic 与真实安全 oracle 的误差有界：

```text
|r_true(E) - r_T(E)| <= eps_cls
```

则：

```text
r_true(E) <= r_t(E) + eps_proxy + eps_cls
```

因此，如果训练目标要求更强的 safety margin：

```text
r_t(E_safe) <= -m
```

且：

```text
m > eps_proxy + eps_cls
```

那么可以推出：

```text
r_true(E_safe) < 0
```

这给出了 target-step 训练的理论条件：

- Z-03 target-step proxy 要和最终风险高度相关；
- 分类器误差越大，需要的训练 margin 越大；
- 如果 `eps_proxy` 很大，只训某一个 step 就可能泛化差，所以 target step ablation 和多 step sampling 是必要实验。

## 8. 命题六：训练收敛性

### 8.1 per-prompt residual 优化

如果不训练 adapter，而是对每个 prompt 直接优化 residual：

```text
min_delta f(delta)
= squared_hinge(r(E) + g^T delta + m)
  + lambda/2 ||delta||^2
```

在局部线性假设下，`f(delta)` 是凸函数；当 `lambda > 0` 时，它是强凸函数。因此存在唯一全局最优解。

若 `f` 是 `L`-smooth、`mu`-strongly convex，用梯度下降：

```text
delta_{k+1} = delta_k - eta grad f(delta_k)
```

当：

```text
0 < eta <= 1/L
```

有线性收敛：

```text
f(delta_k) - f(delta*) <= (1 - eta mu)^k [f(delta_0) - f(delta*)]
```

这可以作为 STG / per-prompt optimization teacher 的理论基础。

### 8.2 adapter 参数训练

神经 adapter：

```text
A_theta(E) = E + Delta_theta(E)
```

整体目标非凸，不能轻易证明全局最优。但可以使用标准 SGD 非凸收敛结论。

设训练目标：

```text
J(theta) = E_{E,t,eps}[L(theta; E,t,eps)]
```

假设：

1. `J(theta)` 下界存在；
2. `J(theta)` 是 `L`-smooth；
3. 随机梯度无偏；
4. 随机梯度方差有界：

```text
E[||g_k - grad J(theta_k)||^2] <= sigma^2
```

则使用合适学习率时，有：

```text
min_{0<=k<K} E[||grad J(theta_k)||^2]
<= O(1 / sqrt(K))
```

这说明 adapter 训练可以收敛到一阶驻点，但不保证全局最优。

如果后续想要更强证明，需要限制 adapter 结构，例如：

- 线性 adapter；
- low-rank linear adapter；
- 固定 token gate + 线性 residual；
- 只学习一个风险子空间投影矩阵。

这类结构更接近 LEACE / INLP，可以给出闭式或凸优化证明。

## 9. 总误差分解

最终可以把方法的安全失败上界写成：

```text
r_true(A_theta(E))
<=
  r_t(E + delta*(E))
  + eps_approx
  + eps_opt
  + eps_proxy
  + eps_cls
  + eps_lin
```

其中：

- `r_t(E + delta*(E))`：理想 per-prompt 最优解在 target-step critic 上的风险；
- `eps_approx`：adapter 无法完全拟合 per-prompt optimum 的摊销误差；
- `eps_opt`：训练没有收敛到最优参数的优化误差；
- `eps_proxy`：target-step `latent_x1` 和最终生成结果风险不一致；
- `eps_cls`：Z-03 critic 和真实人类/强安全 oracle 不一致；
- `eps_lin`：局部线性化近似误差。

如果训练时留出 margin：

```text
r_t(E + delta*(E)) <= -m
```

且：

```text
m >
eps_approx + eps_opt + eps_proxy + eps_cls + eps_lin
```

则可以推出真实风险为安全：

```text
r_true(A_theta(E)) < 0
```

这就是后续最适合写进论文的误差界。它不会夸大方法，但能清楚说明：只要 margin 足够覆盖这些误差源，方法在特定条件下可以保证安全。

## 10. 如何把理论和实验连起来

为了让理论不是空的，建议后续补四组可测量实验。

### 10.1 图像域风险方向和文本 proxy 方向的夹角

估计：

```text
g_img = grad_E C(F_t(E))
g_txt = text-only unsafe direction
```

报告：

```text
cos(g_img, g_txt)
```

如果夹角大，可以支持“文本 proxy 与真实图像域危险空间不一致”的论点。

### 10.2 相关概念误伤和梯度相关性

对目标 IP 和相关概念分别计算：

```text
g_r = grad_E risk_target
g_q = grad_E score_related_or_style
```

报告：

```text
cos(g_r, g_q)
```

如果海绵宝宝和派大星/章鱼哥/卡通风格方向相关性高，就能解释为什么擦除会误伤相关概念。

### 10.3 target-step proxy 误差

比较：

```text
r_t(E) vs r_T(E)
```

统计相关系数和最大误差：

```text
eps_proxy ~= max |r_t - r_T|
```

这能支持当前 `latent_x1` proxy 是否足够可靠。

### 10.4 adapter 摊销误差

对少量 prompt 用 per-prompt gradient search 找到 teacher residual：

```text
delta_teacher(E)
```

比较：

```text
||Delta_theta(E) - delta_teacher(E)||
```

这能估计 `eps_approx`，也能判断 adapter 结构是不是容量不够。

## 11. 推荐论文表述

可以在方法章节这样写：

> Existing prompt-embedding sanitization methods often define unsafe directions in the text embedding space using unsafe tokens or unsafe prompt corpora. However, the text-defined unsafe subspace is only a proxy for the actual unsafe region induced by a text-to-image diffusion model. Since harmful images can be triggered by implicit visual compositions rather than explicit unsafe tokens, this proxy mismatch may leave residual risk or cause collateral semantic damage. We instead define unsafe knowledge through an image-domain latent critic and pull its gradient back through the frozen diffusion trajectory to the prompt embedding space. Under a local linear approximation, the resulting update is the minimum-norm intervention that moves the prompt embedding across the critic-defined safety margin. Our adapter amortizes this per-prompt optimal intervention into a single forward pass.

中文含义：

> 现有 prompt embedding 安全化方法通常用危险词或危险文本语料定义文本空间中的危险方向。然而，这个文本危险子空间只是 diffusion 模型真实危险区域的代理。由于有害图像可能由隐式视觉组合触发，而不一定由显式危险词触发，文本代理空间和真实图像域危险空间之间的 mismatch 会导致残留风险或误伤无关语义。我们用图像域 latent critic 定义危险知识，并通过冻结 diffusion 轨迹把该危险方向拉回到 prompt embedding 空间。在局部线性近似下，这个更新是跨越 critic 安全 margin 的最小范数干预；adapter 则把这种 per-prompt 最优干预摊销成一次前向映射。

## 12. 参考论文

- LEACE: Perfect linear concept erasure in closed form  
  `research/safe_embedding_feedback/papers/LEACE- Perfect linear concept erasure in closed form.pdf`

- INLP: Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection  
  `research/safe_embedding_feedback/papers/Null It Out- Guarding Protected Attributes by Iterative Nullspace Projection.pdf`

- Safe Text-to-Image Generation: Simply Sanitize the Prompt Embedding  
  `research/safe_embedding_feedback/papers/embedding_sanitizer_2411.pdf`

- STG: Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models  
  `research/safe_embedding_feedback/papers/stg_2510.24012.pdf`

- AlignProp: Aligning Text-to-Image Diffusion Models with Reward Backpropagation  
  `research/safe_embedding_feedback/papers/ALIGNING TEXT-TO-IMAGE DIFFUSION MODELS.pdf`

- UCE: Unified Concept Editing in Diffusion Models  
  `research/safe_embedding_feedback/papers/Unified Concept Editing in Diffusion Models.pdf`

- ESD: Erasing Concepts from Diffusion Models  
  `research/safe_embedding_feedback/papers/Erasing Concepts from Diffusion Models.pdf`

- MACE: Mass Concept Erasure in Diffusion Models  
  `research/safe_embedding_feedback/papers/MACE- Mass Concept Erasure in Diffusion Models.pdf`

- Locatello et al.: Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations  
  `research/safe_embedding_feedback/papers/Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.pdf`

- beta-VAE rate-distortion disentanglement  
  `research/safe_embedding_feedback/papers/Understanding disentangling in β-VAE.pdf`

- The Information Bottleneck Method  
  `research/safe_embedding_feedback/papers/The information bottleneck method.pdf`
