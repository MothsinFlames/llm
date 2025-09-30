### ✅ 1. BatchNorm、LayerNorm、RMSNorm 对比

| 特性                  | **BatchNorm**                                          | **LayerNorm**                                     | **RMSNorm**                                                                                                 |
| ------------------- | ------------------------------------------------------ | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **提出时间**            | 2015 (Ioffe & Szegedy)                                 | 2016 (Lei Ba et al.)                              | 2019 (Zhang et al.)                                                                                         |
| **归一化维度**           | 在 batch 维度上（每个 channel）                                | 在特征维度上（每个 token 的 hidden dim）                     | 在特征维度上（每个 token）                                                                                            |
| **统计量**             | 均值 + 方差（per channel per batch）                         | 均值 + 方差（per token）                                | **仅平方均值（RMS）**，无均值                                                                                          |
| **公式**              | \( $\frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$ \) | \($\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$ \) | \( $\frac{x}{\sqrt{\text{RMS}(x)^2 + \epsilon}}$ \)，其中 \($\text{RMS}(x) = \sqrt{\frac{1}{d} \sum x_i^2}$ \) |
| **可学习参数**           | \( \gamma, \beta \)（affine transform）                  | \( \gamma, \beta \)                               | 通常只 \( \gamma \)（scale），无 bias                                                                              |
| **对 batch size 敏感** | ✅ 非常敏感（小 batch 性能差）                                    | ❌ 不敏感                                             | ❌ 不敏感                                                                                                       |
| **适用场景**            | CNN / 视觉任务                                             | Transformer / 大模型                                 | 大模型（LLaMA, Mistral, Qwen 等）                                                                                 |
| **优点**              | 加速 CNN 训练，稳定分布                                         | 适用于动态序列长度、小 batch                                 | 更简单、更快，节省显存                                                                                                 |
| **缺点**              | 不适合 NLP（batch 内长度不一）、不兼容小 batch                        | 多一个可学习 bias（β）                                    | 忽略均值中心化，可能影响某些任务                                                                                            |

---

### ✅ 2. 大模型中常见的其他归一化方法

#### 🔹 **1. DeepNorm / DeepNet-style Normalization**
- **动机**：训练极深 Transformer（如 100+ 层）时稳定训练。
- **做法**：调整残差连接和归一化的顺序与缩放。
  - 例如：**Pre-normalization + 残差缩放**（如 `x + α·f(LN(x))`）。
- **代表模型**：DeepNet、HuggingFace 的 *BART* 使用类似结构。

#### 🔹 **2. ScaleNorm**
- 简化版 LayerNorm：不学 `γ`，而是用一个标量 `g` 缩放整个输出。
- 公式：\( \text{ScaleNorm}(x) = g \cdot \frac{x}{\|x\|} \)
- 更少参数，适合某些稳定训练场景。

#### 🔹 **3. T5 LayerNorm（+ no bias）**
- T5 使用标准 LayerNorm，但：
  - **去掉 bias 项**（即 β = 0）。
  - **初始化 γ = 0**（在某些 FFN 输出前）。
- 目的：提升训练稳定性，控制梯度爆发。

#### 🔹 **4. ALiBi (Attention with Linear Biases) + RMSNorm**
- 虽然 ALiBi 是位置编码机制，但它与 RMSNorm 搭配良好（如 **Yi, Falcon, MPT**）。
- 特点：**完全不需要位置嵌入**，RMSNorm 配合线性注意力偏置。

#### 🔹 **5. Q-Normalization（近期研究，如 LLM.int8()）**
- 用于量化训练或低精度场景。
- 对 Query 在归一化前做归一化，防止注意力 softmax 溢出。

#### 🔹 **6. SublayerNorm / iRope & QKNorm（如 Qwen-2.5）**
- **QKNorm**：对 Q 和 K 单独归一化（如 RMS），增强注意力稳定性。
- **SublayerNorm**：在每个子层（如 attention 或 FFN）内部使用轻量归一化。

---

### ✅ 3. 当前大模型趋势总结（2023–2025）

| 归一化方法                 | 使用模型                               | 说明                           |
| --------------------- | ---------------------------------- | ---------------------------- |
| **RMSNorm**           | LLaMA, Mistral, Qwen, Yi, DeepSeek | 主流选择，简洁高效                    |
| **LayerNorm**         | BERT, T5, BART                     | 传统 Transformer 标配，逐渐被 RMS 取代 |
| **RMSNorm + QKNorm**  | Qwen-2.5, Qwen-3                   | 提升注意力稳定性                     |
| **No Bias in Norm**   | T5, LLaMA（FFN output）              | 减少参数，提升泛化                    |
| **Pre-LN + DeepNorm** | DeepNet, GLM                       | 极深网络训练稳定                     |
