# Safe Embedding Feedback 研究报告

日期：2026-07-15

## 结论

这个方向可行，但不建议直接上纯 `flow-GRPO`。更稳的路线是：

1. 先把“危险 prompt -> 安全 embedding”做成一个轻量 adapter；
2. 用现有 diffusion 反馈做 reward，先做监督/伪标签预热；
3. 再用可微 reward 或小规模在线 RL 微调；
4. `flow-GRPO` 只作为后期在线优化手段，不作为第一版主训练框架。

原因很直接：

- 你已经有可用的 Z-03 latent classifier 入口，且仓库里现成支持 `prompt_embeds`；
- STG 这类方法已经证明了“在采样过程中改文本 embedding”是有效的；
- 纯 RL 在这个问题上很容易 reward hacking 和语义漂移；
- 单一 latent classifier 适合做局部安全信号，不适合单独承担全部安全目标。

## 和当前代码库的关系

这个仓库里已经有很接近你目标的基础设施：

- `pipelines/ZImageSTGPipeline.py`：现成的 prompt embedding 迭代更新框架。
- `pipelines/ZImageIPPipeline.py`：支持 `prompt_embeds`，适合接入 learned safe embedding。
- `latent_vit/infer_z03_latent.py`：Z-03 latent 安全分类器的原生推理入口，直接输出 porn/gore/ip 三个分类头。
- `adapter/latent_vit_ip_detector.py`：基于 Z-03 backbone 的特征检索封装，不是 latent 分类器本身的主用法。
- `run_zimage_latent_vit_ip_detect.py`：特征检索式 IP 检测入口和 reference bank 构建器。
- `adapter/clip_safety_model.py`：可以作为额外 reward / safety critic。

这意味着你的方法不需要从零搭 diffusion 训练链路，主要工作是把“打分器”改成可训练 reward，把“在线改 embedding”改成一个学生模型。

## 相关论文

已下载或纳入调研的核心论文：

- `stg_2510.24012.pdf` - Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models
- `peo_2510.02599.pdf` - Prompt Embedding Optimization
- `drtune_2405.00760.pdf` - Deep Reward Supervisions for Tuning Text-to-Image Diffusion Models
- `dpok_2305.16381.pdf` - DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models
- `diffexp_2502.14070.pdf` - DiffExp: Efficient Exploration in Reward Fine-tuning for Text-to-Image Diffusion Models
- `flow_grpo_2505.05470.pdf` - Flow-GRPO: Training Flow Matching Models via Online RL

参考但未落盘：

- AlignProp: Aligning Text-to-Image Diffusion Models with Reward Backpropagation

这些论文给出的共同信号是：把 reward 从最终图像传回到扩散轨迹，或者直接把 prompt embedding 当优化变量，是成立的。

## 论文启发到方案组件

STG 证明了安全函数可以直接作用到文本 embedding 上。它不是训练一个 adapter，而是在采样时迭代更新 embedding；这很适合作为 teacher，产出 `(dangerous_embedding, safe_embedding)` 训练对。

PEO 证明了 prompt embedding optimization 不只适用于安全，也能优化审美等软目标。它支持一个关键判断：embedding 空间本身是可优化控制面，不一定要微调整个 diffusion backbone。

AlignProp / DRTune 证明了 reward 可以通过 denoising process 反传。DRTune 还强调中间过程监督的价值，这和当前仓库的 Z-03 latent 分类器非常契合。

DPOK / DiffExp 说明在线 reward fine-tuning 可行，但需要处理探索、样本效率和 KL 约束。它们更像第三阶段工具，不适合第一版直接替代监督预热。

Flow-GRPO 说明 flow matching 模型可以接在线 RL，但它主要面向模型参数对齐。对你的问题，如果只训练 embedding adapter，先用更简单的 group reward / KL 更新即可；只有当你要调整 flow/diffusion backbone 时，才需要完整 Flow-GRPO 机制。

## 可行性判断

### 可行

- 输入危险 prompt embedding，输出安全 embedding，这个形式天然适合当前 Z-Image 管线。
- 你的 Z-03 latent classifier 可以充当 reward 或 reward proxy。
- 如果把 STG 生成的安全 embedding 当 teacher，先做监督蒸馏，训练会稳定很多。

### 风险

- 只靠 Z-03 latent classifier，安全覆盖面会受训练标签限制，尤其是标签外风险。
- 只优化一个 embedding 向量，容易损失语义和可控性。
- 直接上 GRPO 类在线优化，样本效率和稳定性都不理想。
- 只看中间 latent，可能和最终图像安全存在偏差。

### 建议

- 用 token-wise residual embedding，而不是单个 pooled vector。
- reward 至少包含三部分：安全、语义保持、图像质量。
- 先离线蒸馏，再在线强化。
- 对安全 critic 做多头化，不要只靠单一 Z-03 输出头。

## 推荐实验顺序

1. 先复现 STG / PEO 风格的 safe embedding 轨迹。
2. 用这些轨迹训练一个 student adapter。
3. 把 Z-03 porn/gore/ip logits 改成可微 reward，做小步 fine-tune。
4. 再尝试 flow-GRPO / group RL。
5. 最后做跨攻击方式、跨 prompt 风格的鲁棒性评估。

## 第一版不要做的事

- 不要直接训练整个 Z-Image backbone，代价高且难排查。
- 不要只用 hard threshold 当 reward，梯度信息太少。
- 不要只优化安全分数，必须同时约束语义和 embedding 范数。
- 不要让 adapter 在 benign prompts 上强行改写，否则 false positive 会很高。

## 初步结论

这条路线值得做，而且有现成代码基础。最合理的版本不是“一个 embedding-to-embedding 模型替代一切”，而是“学生 adapter + 可微 reward + 受约束的在线优化”。
