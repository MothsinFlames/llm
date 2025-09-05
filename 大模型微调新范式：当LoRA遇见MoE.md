---
created: 2025-08-20T14:18:22 (UTC +08:00)
tags: [大模型,人工智能,ChatGРТ]
source: https://zhuanlan.zhihu.com/p/683637455
author: 北京大学 计算机科学与技术硕士
---

# 大模型微调新范式：当LoRA遇见MoE

> ## Excerpt
> 引言当LoRA遇见MoE，会擦出怎样的火花？ 由于大模型全量微调时的显存占用过大，LoRA、Adapter、IA ^3 这些参数高效微调（Parameter-Efficient Tuning，简称PEFT）方法便成为了资源有限的机构和研究者微调大模型的…

---
## 引言

当LoRA遇见MoE，会擦出怎样的火花？

![](https://pic3.zhimg.com/v2-0ee044476a0f4506477f8977c50e0f90_1440w.jpg)

左侧：原始版本的LoRA，权重是稠密的，每个样本都会激活所有参数；右侧：与混合专家（MoE）框架结合的LoRA，每一层插入多个并行的LoRA权重（即MoE中的多个专家模型），路由模块（Router）输出每个专家的激活概率，以决定激活哪些LoRA模块。

由于大模型全量微调时的显存占用过大，LoRA、Adapter、IA $^3$ 这些**参数高效微调**（Parameter-Efficient Tuning，简称**[PEFT](https://zhida.zhihu.com/search?content_id=240049673&content_type=Article&match_order=1&q=PEFT&zhida_source=entity)**）方法便成为了资源有限的机构和研究者微调大模型的标配。PEFT方法的总体思路是冻结住大模型的主干参数，引入一小部分可训练的参数作为适配模块进行训练，以节省模型微调时的显存和参数存储开销。

传统上，LoRA这类适配模块的参数和主干参数一样是**稠密**的，每个样本上的推理过程都需要用到所有的参数。近来，大模型研究者们为了克服稠密模型的参数效率瓶颈，开始关注以[Mistral](https://zhida.zhihu.com/search?content_id=240049673&content_type=Article&match_order=1&q=Mistral&zhida_source=entity)、[DeepSeek MoE](https://zhida.zhihu.com/search?content_id=240049673&content_type=Article&match_order=1&q=DeepSeek+MoE&zhida_source=entity)为代表的混合专家（Mixure of Experts，简称**MoE**）模型框架。在该框架下，模型的某个模块（如Transformer的某个[FFN层](https://zhida.zhihu.com/search?content_id=240049673&content_type=Article&match_order=1&q=FFN%E5%B1%82&zhida_source=entity)）会存在多组形状相同的权重（称为**专家**），另外有一个**路由模块（Router）**接受原始输入、输出各专家的激活权重，最终的输出为：

-   如果是**软路由（soft routing**），输出各专家输出的加权求和；
-   如果是**离散路由（discrete routing ）**，即Mistral、DeepDeek MoE采用的稀疏混合专家（Sparse MoE）架构,则将Top-K（K为固定的 超参数，即每次激活的专家个数，如1或2）之外的权重置零，再加权求和。

在MoE架构中，每个专家参数的激活程度取决于数据决定的路由权重，使得各专家的参数能各自关注其所擅长的数据类型。在离散路由的情况下，路由权重在TopK之外的专家甚至不用计算，在保证总参数容量的前提下极大降低了推理的计算代价。

那么，对于已经发布的稠密大模型的PEFT训练，是否可以应用MoE的思路呢？近来，笔者关注到研究社区开始将以LoRA为代表的PEFT方法和MoE框架进行结合，提出了**MoV**、**MoLORA**、**LoRAMOE**和**MOLA**等新的PEFT方法，相比原始版本的LORA进一步提升了大模型微调的效率。

本文将解读其中三篇具有代表作的工作，以下是太长不看版：

1.  [MoV和MoLORA](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2309.05444) \[1\]：提出于2023年9月，首个结合PEFT和MoE的工作，MoV和MoLORA分别是IA $^3$ 和LORA的MOE版本，采用token级别的软路由（加权合并所有专家的输出）。作者发现，对3B和11B的T5大模型的SFT，MoV仅使用不到1%的可训练参数量就可以达到和全量微调相当的效果，显著优于同等可训练参数量设定下的LoRA。
2.  [LoRAMOE](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2312.09979) \[2\]：提出于2023年12月，在MoLORA \[1\]的基础上，为解决微调大模型时的灾难遗忘问题，将同一位置的LoRA专家分为两组，分别负责保存预训练权重中的世界知识和微调时学习的新任务，并为此目标设计了新的负载均衡loss。
3.  [MOLA](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2402.08562) \[3\]：提出于2024年2月，使用离散路由（每次只激活路由权重top-2的专家），并发现在每一层设置同样的专家个数不是最优的，增加高层专家数目、降低底层专家数目，能在可训练参数量不变的前提下，明显提升LLaMa-2微调的效果。

## MoV和MoLORA：PEFT初见MoE，故事开始

论文链接：[Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient MoE for Instruction Tuning](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2309.05444)

该工作首次提出将LoRA类型的PEFT方法和MoE框架进行结合，实现了MoV（IA $^3$ 的MOE）版本和MoLORA（LORA的MOE）版本，发现MoV的性能在相等的可训练参数量设定下优于原始的LORA，非常接近全参数微调。回顾下IA $^3$ 的适配模块设计，即在Transformer的K、V和FFN的第一个全连接层的输出，各自点乘上一个可训练的向量 $l_{\text{k}}, l_{\text{v}}, l_{\text{ff}}$ ：

$\operatorname{softmax}\left(\frac{Q\left(l_{\mathrm{k}} \odot K^T\right)}{\sqrt{d_{\mathrm{k}}}}\right)\left(l_{\mathrm{v}} \odot V\right) ; \quad\left(l_{\mathrm{ff}} \odot \gamma\left(W_1 x\right)\right) W_2.$

那么，MOV就是将这些可训练向量各自复制 $n$ 份参数（n为专家个数），并加入一个路由模块接受K/V/FFN原本输出的隐向量 $x_{\text{hidden}}$ 、输出各专家的激活权重，过softmax之后得到各专家的激活概率 $s$ ，以其为权重对各个可训练向量求和（之后向原始版本的IA $^3$ 一样，将求和后的向量点乘上原输出的隐向量）。图示如下：

![](https://picx.zhimg.com/v2-8bdad4f931025baea38bb4a981ec73b5_1440w.jpg)

MOV方法的示意图，引自论文\[1\]。

实验部分，作者在Public Pool of Prompts数据集上指令微调了参数量从770M到11B的T5模型，在8个held out测试集上进行测试。实验的微调方法包括全量微调、原始版本的IA $^3$ 和LORA、MoV和MoLORA。从测试结果上来看，**MoV的性能明显好于MoLORA和原始版本的IA** $^3$ **和LORA**。例如，专家个数 $n=10$ 的MoV-10只用3B模型0.32%的参数量，就能达到和全量微调相当的效果，明显优于同等可训练参数量的IA $^3$ 和LORA，而使用0.68%可训练参数的MoV-30（60.61）甚至超过全量微调。

![](https://pic3.zhimg.com/v2-1ea83107ce24a6ba9dcadaa011748806_1440w.jpg)

3B模型的测试结果，只使用0.32%可训练参数的MoV-10的平均accuracy（59.93）接近全量微调（60.06），明显优于使用0.3%可训练参数的原始版本LORA（57.71）。使用0.68%可训练参数的MoV-30（60.61）甚至超过全量微调。

此外，作者还对专家的专门程度（speciality，即每个任务依赖少数几个特定专家的程度）进行了分析，展示MOV-5微调的770M模型最后一层FFN中各专家路由概率的分布：

![](https://pic1.zhimg.com/v2-7c2ddcbf8bc9394bd349d6f9a090b8b6_1440w.jpg)

路由概率的分布，左侧为模型在训练集中见过的任务，右侧为测试集中模型未见过的任务。

可以看出，无论模型是否见过任务数据，大多数任务都有1-2个特别侧重的专家占据了大部分激活概率值，说明MoV这个MoE实现达成了专家的专门化。

## LoRAMOE：LoRA专家分组，预训练知识记得更牢

论文链接：[LoRAMoE: Revolutionizing Mixture of Experts for Maintaining World Knowledge in Language Model Alignment](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2312.09979)

此文为复旦大学NLP组的工作，研究动机是解决**大模型微调过程中的灾难遗忘问题。**

作者发现，随着所用数据量的增长，SFT训练会导致**模型参数大幅度偏离预训练参数**，预训练阶段学习到的**世界知识（world knowledge）**逐渐被遗忘，虽然模型的指令跟随能力增强、在常见的测试集上性能增长，但需要这些世界知识的QA任务性能大幅度下降：

![](https://pic3.zhimg.com/v2-de86d811dfded17a66e6f642a2e7d780_1440w.jpg)

左侧为不需要世界知识的常见测试集上的性能，右侧为需要世界知识的QA测试集上的表现，横轴为SFT数据量，红线为模型参数的变化程度。

作者提出的解决方案是：

-   **数据部分：**加入world knowledge的代表性数据集CBQA，减缓模型对世界知识的遗忘；
-   **模型部分：**以（1）减少模型参数变化、（2）隔离处理世界知识和新任务知识的参数为指导思想，在上一篇文章的MoLORA思想上设计了LoRAMoE方法，将LoRA专家们划分为两组，一组用于保留预训练参数就可以处理好的（和世界知识相关的）任务，一组用于学习SFT过程中见到的新任务，如下图所示：

![](https://pica.zhimg.com/v2-021909244a00b75d1831ae543b954430_1440w.jpg)

为了训练好这样的分组专家，让两组专家在**组间各司其职**（分别处理两类任务）、在**组内均衡负载**，作者设计了一种名为**localized balancing contraint**的负载均衡约束机制。具体地，假设 $\mathbf{Q}$ 为路由模块输出的重要性矩阵， $\mathbf{Q}_{n,m}$ 代表第 $n$ 个专家对第 $m$ 个训练样本的权重， $\mathbf{I}$ 为作者定义的和 $Q$ 形状相同的矩阵：

$\mathbf{I}_{n, m}=\left\{\begin{array}{c} 1+\delta, \quad \operatorname{Type}_e(n)=\operatorname{Type}_s(m) . \\ 1-\delta, \quad \operatorname{Type}_e(n) \neq \operatorname{Type}_s(m) \cdot 1 \end{array}\right.$

其中 $\delta$ 为0-1之间的固定值（控制两组专家不平衡程度的超参）， $\operatorname{Type}_e(n)$ 为第 $n$个专家的类型（设负责保留预训练知识的那组为0，负责学习新任务的那组为1）， $\operatorname{Type}_s(m)$ 为第 $m$ 个样本的类型（设代表预训练知识的CBQA为0，其他SFT数据为1）。负载均衡损失 $\mathcal{L}_{lbc}$ 的定义为用 $\mathbf{I}$ 加权后的重要性矩阵 $\mathbf{Z}=\mathbf{I} \circ \mathbf{Q}$ 的方差除以均值：

$\mathcal{L}_{l b c}=\frac{\sigma^2(\mathbf{Z})}{\mu(\mathbf{Z})}$ .

这样设计loss的用意是，对任意一种训练样本，两组LoRA专家组内的 $\mathbf{I}$ 值是相等的，优化 $\mathcal{L}_{l b c}$ 即降低组内路由权重的方差，使得组内负载均衡；两组专家之间，设专家组A对当前类型的数据更擅长，则其 $\mathbf{I}$ 值大于另一组专家B，训练起始阶段的A的激活权重就显著大于B，A对这种数据得到的训练机会更多，路由模块在训练过程中逐渐更倾向对这种数据选择A组的专家。这种**“强者愈强”的极化现象**是MoE领域的经典问题，可以参见经典的sMoE论文[The Sparsely-Gated Mixture-of-Experts Layer](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1701.06538) \[4\]对该问题的阐述。

这样一来，即使推理阶段没有数据类型 $\mathbf{I}$的信息，A对这种数据的路由值 $\mathbf{Q}$ 也会显著大于B的相应值，这就实现了两组专家各司其职的目标。

实验部分，作者在CBQA和一些列下游任务数据集混合而成的SFT数据上微调了LLaMA-2-7B，对比了全量SFT、普通LORA和作者所提的LoRAMoE的性能。结果显示，LoRAMoE有效克服了大模型SFT过程中的灾难性遗忘问题，在需要世界知识的QA任务（下表下半部分）上性能最佳，在与SFT训练数据关系更大的其他任务上平均来说基本与SFT训练的模型相当：

![](https://pica.zhimg.com/v2-d127990b2f743340d66bf54588b3b560_1440w.jpg)

## MOLA：统筹增效，更接近输出端的高层需要更多专家

论文链接：[Higher Layers Need More LoRA Experts](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2402.08562.pdf)

该工作受到MoE领域先前工作\[5\]发现的**专家个数过多容易导致性能下降**的现象之启发，提出了两个问题：

1.  现有PEFT+MoE的微调方法是否存在**专家冗余**的问题？
2.  如何在不同中间**层之间分配专家个数**？

为了解答问题1，作者训练了每层专家个数均为5的LoRA+MoE（基座模型为32层的LLaMa-2 7B），路由机制采用Top-2离散路由，计算了每层self-attention的Q、K、V、O各组专家权重内两两之间求差的Frobenius范数的平均值，可视化如下：

![](https://pic4.zhimg.com/v2-8c5ba58fbbb8883a8711a7799ad71889_1440w.jpg)

横轴为模型层数，纵轴为专家权重之间的差异程度。

可以看出，层数越高（约接近输出端），专家之间的差异程度越大，而低层的专家之间差异程度非常小，**大模型底层的LoRA专家权重存在冗余**。该观察自然导出了对问题2答案的猜想：**高层需要更多专家**，在各层的专家个数之和固定的预算约束下，应该把底层的一部分专家挪到高层，用原文标题来说就是：

> Higher Layers Need More Experts

为了验证该猜想，作者提出了四个版本的专家个数划分方式分别严重性能，它们统称为**MoLA** （**M**oE-L**o**RA with **L**ayer-wise Expert **A**llocation），分别是：

-   MoLA-△：正三角形，底层专家个数多，高层专家个数少；
-   MoLA-▽：倒三角形，底层少，高层多；
-   MoLA-▷◁: 沙漏型，两头多、中间少；
-   MoLA-□：正方形，即默认的均匀分配。

![](https://pic2.zhimg.com/v2-f4956fea80ba0fcd5650f861b225e465_1440w.jpg)

四种在不同中间层之间划分专家个数的方式。

具体实现中，作者将LLaMA的32层从低到高分为4组，分别是1-8、9-16、17-24、25到32层，以上四种划分方式总的专家个数相等，具体划分分别为：

-   MoLA-△：8-6-4-2
-   MoLA-▽：2-4-6-8；
-   MoLA-▷◁: 8-2-2-8；
-   MoLA-□：5-5-5-5。

路由机制为token级别的Top-2路由，训练时加入了负载均衡损失。MoLA的LoRA rank=8，基线方法中LoRA的秩为64（可训练参数量略大于上述四种MoLA，与MOLA-□的8-8-8-8版本相同）评测数据集为MPRC、RTE、COLA、ScienceQA、CommenseQA和OenBookQA，在两种设定下训练模型：

-   设定1：直接在各数据集的训练集上分别微调模型；
-   设定2：先在OpenOrac指令跟随数据集上进行SFT，再在各数据集的训练集上分别微调模型。

从以下实验结果可以看出，在设定1下，MoLA-▽都在大多数数据集上都取得了PEFT类型方法的最佳性能，远超可训练参数量更大的原始版本LoRA和LLaMA-Adapter，相当接近全量微调的结果。

![](https://pic1.zhimg.com/v2-8c1d7dc12cb4de7f2e86372ae00b906a_1440w.jpg)

设定1下的实验结果

在设定2下，也是倒三角形的专家个数分配方式MoLA-▽最优，**验证了“高层需要更多专家”的猜想。**

![](https://pic3.zhimg.com/v2-73c35519c5436e80821e6d78991c4ac6_1440w.jpg)

笔者点评：从直觉来看，模型的高层编码更high-level的信息，也和目标任务的训练信号更接近，和编码基础语言属性的底层参数相比需要更多调整，和此文的发现相符，也和迁移学习中常见的layer-wise学习率设定方式（顶层设定较高学习率，底层设定较低学习率）的思想不谋而合，未来可以探索二者的结合是否能带来进一步的提升。

___

我是

，一枚即将从北大毕业的NLPer，日常更新LLM和深度学习领域前沿进展，也接算法面试辅导，欢迎关注和赐读往期文章，多多交流讨论：

![](https://pica.zhimg.com/v2-3d8c6e6065a647e10bab26a8557f86f7_l.jpg?source=f2fdee93)

Sam聊算法

19 次咨询

5.0

15156 次赞同

去咨询

#大模型 #大语言模型 #混合专家模型 #参数高效微调 #LLM #人工智能 #深度学习 #自然语言处理 #NLP #模型加速 #论文分享 #算法面试

**参考文献**

\[1\] [Zadouri, Ted, et al. "Pushing mixture of experts to the limit: Extremely parameter efficient moe for instruction tuning."arXiv preprint arXiv:2309.05444(2023).](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2309.05444.pdf)

\[2\] [Dou, Shihan, et al. "Loramoe: Revolutionizing mixture of experts for maintaining world knowledge in language model alignment."arXiv preprint arXiv:2312.09979(2023).](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2312.09979)

\[3\] [Gao, Chongyang, et al. "Higher Layers Need More LoRA Experts."arXiv preprint arXiv:2402.08562(2024).](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2402.08562)

\[4\] [Shazeer, Noam, et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer."International Conference on Learning Representations. 2016.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1701.06538)

\[5\] [Chen, Tianlong, et al. "Sparse MoE as the New Dropout: Scaling Dense and Self-Slimmable Transformers."The Eleventh International Conference on Learning Representations. 2022.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2303.01610)
