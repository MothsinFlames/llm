---
created: 2025-08-20T15:52:33 (UTC +08:00)
tags: [预训练模型,生成模型,大规模预训练模型]
source: https://zhuanlan.zhihu.com/p/406751681
author: 关于作者guolipa工作学习一会就瞌睡郭达森也关注了他回答13文章11关注者727关注他发私信
---

# 生成式预训练模型：UniLM、BART、T5、GPT

> ## Excerpt
> BERT等预训练模型通过在大规模无标注的文本语料上进行预训练，获得了具有强大表达能力和泛化性的文本编码器，在自然语言理解任务（如文本分类、阅读理解、信息抽取）上取得了显著的性能效果。但NLP中还有另一大类…

---
BERT等预训练模型通过在大规模无标注的文本语料上进行预训练，获得了具有强大表达能力和泛化性的文本编码器，在自然语言理解任务（如文本分类、阅读理解、信息抽取）上取得了显著的性能效果。但NLP中还有另一大类任务——自然语言生成，如机器翻译、文本摘要、对话生成等。而这类任务不仅需要对输入文本进行较好的表示编码，还需要一个强大的解码器生成文本。因此，针对文本生成任务，一些列相关的生成式预训练模型提出了。

## 1\. BERT

首先这部分对BERT的预训练方式进行简单介绍。BERT采用的**预训练（pre-training）- 微调（fine-tuning）范式是后续许多预训练模型的基本模式。**

-   **预训练：**在不同任务组成的大规模无标记自然语言数据集上进行无监督的预训练
-   **微调：**利用预训练的模型在具体的下游任务上使用标注的数据对模型进行有监督的微调

![](https://picx.zhimg.com/v2-044c7289879a2a65fdb155bd16b88757_1440w.jpg)

BERT预训练和微调示意图

BERT使用了以下两种无监督的任务目标来对模型进行预训练：

-   **掩盖语言模型（Masked Language Model，MLM）**

为了训练双向的深度token表示，将输入文本中一定比例的tokens进行掩盖（mask），并预测这些mask的tokens。在论文实验中，每个输入文本序列掩盖15%的tokens，同时这15%需要掩盖的tokens中80%进行真实的掩盖，10%随机的替换成其他的token，10%保持原词不变。

-   **下一个句子预测（Next Sentence Prediction， NSP）**

这一任务是为了训练模型理解句子间关系，训练语料可以从语料库中抽取包括两个句子A和B的句子对进行生成，其中50%的概率B是A的下一个句子，50%的概率B是语料中的一个随机句子，NSP任务预测B是否是A的下一句。

GPT由OpenAI公司从2018年开始陆续提出的一系列预训练模型，目前一共有三个版本：GPT-1、GPT-2和GPT-3，不同版本的GPT模型结构相差不大，但是模型参数规模却不断变大，比如GPT-3就有1750亿个参数，是GPT-2的100倍，性能也逐渐变得强大，支持few-shot、one-shot和zero-shot等下游任务。

> **GPT-1**: [Improving Language Understanding by Generative Pre-Training, 2018](https://link.zhihu.com/?target=https%3A//cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
> 
> **GPT-2**: [Language Models are Unsupervised Multitask Learners, 2019](https://link.zhihu.com/?target=https%3A//cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
> 
> **GPT-3**: [Language Models are Few-Shot Learners, 2020](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2005.14165.pdf)

### 2.1 GPT-1

GPT-1的提出要早于BERT模型，也是采用“预训练-微调”的模式，在大规模无标记的文本语料上进行无监督的预训练，然后再在特定任务上进行有监督的微调，这种范式与BERT是一致的。

![](https://pica.zhimg.com/v2-27ac42e40931bb59c225590a795e0a22_1440w.jpg)

GPT-1的整体模型结构，以及在不同任务上微调示意图

**与BERT的双向掩码语言模型不同，GPT是自回归语言模型。BERT由Transformer的Encoder部分堆叠组成，而GPT使用的是Transformer的Decoder部分，更适合于文本生成任务。**给定输入文本token序列 $X=\{x_{1}, x_{2},...,x_{n}\}$ ，GPT使用标准的语言模型目标函数进行优化： $L(X) = \sum_{i}{logP(x_{i}|x_{1},...,x_{i-1})}$

![](https://pic4.zhimg.com/v2-43e9ec5ab8253d35d5755c6c4f14931d_1440w.jpg)

### 2.2 GPT-2

GPT-2模型结构和GPT-1相同是自回归语言模型，仍然使用Transformer的Decoder组成，预训练使用的数据以及模型参数规模但相比GPT-1变得更大，GPT-2模型参数规模大约是GPT-1的10倍左右，同时GPT-2采用多任务学习的预训练，对于下游任务主打zero-shot，不再需要微调即可使用。

![](https://pic1.zhimg.com/v2-557f00012ad48bdc7c13d900119b8a34_1440w.jpg)

不同规模的模型超参数，最小的11M是GPT-1，最大的1542M是GPT-2，参数规模相差一个量级

GPT-2在预训练时是采用多任务学习的方式，即同一个模型在多个任务上进行无监督训练和收敛，进一步提升模型的泛化能力。但是应用到每一个下游任务微调时，都需要有标签的数据，并且要在预训练模型基础上重新训练task-specific模型。因此，**GPT-2提出在zero-shot的设定下语言模型能够执行下游任务，不需要下游任务的任何标注信息，也不需要重新训练模型**，修改任何模型参数和结构，即预训练一个模型，在多个任务上都能直接用。

在执行下游任务时，不同于GPT-1给定输入直接输出的条件概率分布 $p(output|input)$ ，为了在下游任务实现zero-shot，GPT-2根据给定的输入文本和任务提示信息进行输出，即$p(output|input,task)$。具体来说就是，**在用预训练好的GPT-2执行特定任务时，只需要在输入文本之前添加任务相关的提示信息，告诉模型要执行哪类任务，模型就能直接处理输出**，这也就是当前最火的**[Prompt机制](https://zhida.zhihu.com/search?content_id=178519752&content_type=Article&match_order=1&q=Prompt%E6%9C%BA%E5%88%B6&zhida_source=entity)**。比如：

-   机器翻译任务：(translate to french, english text) ---> (french text)
-   阅读理解任务：(answer the question, document, question) ---> (answer)

![](https://pic2.zhimg.com/v2-77c0fb736e5003beb8c7bfaad724397b_1440w.jpg)

在多个数据集任务上GPT-2的zero-shot实验结果

从上图的实验结果来看，模型越大，性能也越好，而且此时最大的GPT-2性能还没有到达极限，同时在zero-shot设定下的性能效果并非很理想。因此，后面更大参数规模、面向不同任务设定的GPT-3也就应运而生。

### 2.3 GPT-3

GPT-3延续了GPT-2的单向Transformer的自回归语言模型结构，但将模型参数规模是GPT-2的100倍，1750亿个参数。GPT-3不在追求zero-shot的设定，而是在下游任务中给定少量标注的样本让模型学习再进行推理生成。因此，**GPT-3主要展示了超大规模语言模型的小样本学习能力**。

-   **模型思路动机**

现有的预训练-微调的模型还存在局限性：首先，每个新任务都需要大量标记样例的数据集进行微调，限制了语言模型的适用性；其次，模型在大规模训练数据上可能过拟合，而在数据规模小且分布很狭窄的新任务上泛化能力差；最后，人类在学习大多数任务时不需要大规模的监督数据。

解决上述问题的思路有两个：

**① 扩大模型（Large-scale Transformer）**：这个很好理解，模型越大能容纳学习的知识越多，性能越好，同时训练大模型也需要更大规模的文本数据；

**② 元学习 (Meta Learning)**：模型的无监督预训练过程可以看做从很多不同任务中进行元学习的过程。如下图所示，每个任务可看做一个序列，每个序列包含多个连续的具体任务样本，模型在该序列上的训练则称为内循环 (Inner loop)，也称为 In-Context Learning。模型在不通序列上的训练对应元学习的外循环，提升模型的泛化能力，避免在某个特定任务上过拟合。

![](https://pic4.zhimg.com/v2-93ce1d7770c42effb3d71ed6f3647aed_1440w.jpg)

-   **模型使用**

GPT-3经过预训练后不需要微调直接应用于下游任务，而且应用时，模型的输入不仅包括文本和task-specfic前缀，还使用了少量的目标任务标注样本作为上下文条件。同时作者还对比了Zero-shot、One-shot、Few-shot和Fine-tuning四种类型的任务设置，不同类型的任务需要的标注样本是不同的。例如，对于机器翻译任务，对于Zero-shot情况，只需要构建包含任务描述和文本提示输入模型，对One-shot情况，还需要在任务描述之后添加一个标注的样本，而对Few-shot，就需要添加几个标注的样本，如下图所示：

![](https://pica.zhimg.com/v2-960b424ca59b1daa38c38d0b2cbf46ae_1440w.jpg)

论文通过大量的实验证明，在zero-shot、one-shot 和few-shot设置下，GPT-3 在许多 NLP 任务和基准测试中表现出强大的性能。**GPT-3模型不需要任何额外的微调，就能够在只有少量目标任务标注样本的情况下进行很好的泛化，再次证明大力出击奇迹，做大模型的必要性。**

## 3\. UniLM

微软提出的UniLM (**Uni**fied pre-trained **L**anguage **M**odel )是可以同时针对自然语言理解 (NLU) 和自然语言生成 (NLG) 任务进行微调的预训练模型。UniLM使用三种类型的语言建模任务进行预训练：**单向语言模型 (unidirectional LM)、双向语言模型 (bidirectional LM) 、序列到序列语言模型 (sequence-to-sequence LM)** 。

![](https://picx.zhimg.com/v2-feb2c4f2b8fd23bc26233b478be6dc67_1440w.jpg)

UniLM模型结构图

**3.1 Backbone Network: Multi-Layer Transformer**

和BERT一样，UniLM模型是多层Transformer组成。给定输入文本，在输入开始添加 $[SOS]$ ，在每个segment的结尾添加 $[EOS]$ ，e.g., $[SOS]x_{1} x_{2}[EOS]x_{3}x_{4}x_{5}[EOS]$ 。 输入中每个token的向量通过对应的token embedding、position embedding、segment embedding相加得到，传入 $L$ 层的Transformer进行上下文编码建模， $H^{l}=Transformer _{l}(H^{l-1})$ 。每一层自注意力头的输出 $A^{l}$ 通过如下公式计算：

![](https://pic3.zhimg.com/v2-81b07c60a3728afc88da5eb3890d18ca_1440w.jpg)

不同类型的语言模型使用不同的掩码矩阵 $M$ 来控制一个token计算其隐层表示时能够和哪些上下文token交互，上图中展示了三种不同类型的掩码方式。

**3.2 Pre-training Objectives**

-   **\[Unidirectional LM\]:** 论文同时用left-to-right和right-to-left的语言模型目标函数，对于left-to-right模型，输入 $"x_{1}x{2}[MASK]x_{4}"$ ，预测 $[MASK]$ 位置token时，只有其本身和左侧的token可以使用。模型的掩码矩阵 $M$ 是一个三角形矩阵，其中上三角部分值设置为 $-\infty$ ，其他值为 $0$ 。
-   **\[Bidirectional LM\]:** 双向语言模型和BERT一样，每个token可以和输入序列中任意token交互，掩码矩阵$M$中的值均为 $0$ 。
-   **\[Sequence-to-Sequence LM\]:** Seq2Seq语言模型输入包括两个segment，source segment和target segment，e.g., $[SOS]s_{1} s_{2}[EOS]t_{1}t_{2}t_{3}[EOS]$ 。在建模过程中，source中的token之间可以随意交互，target中的token可以与其左侧的token以及source中的token交互。在预训练时，在source和target两个部分同时随机选择token替换为\[MASK\]，利用模型对掩盖的部分进行预测恢复。在微调生成任务时，则只对target部分的token(包括 $[EOS]$ )进行随机掩盖预测。而用微调好的模型进行文本生成时，只给定source部分，预测生成target部分。
-   **\[Next Sentence Prediction\]:** 和BERT中的下一个句子预测任务相同。

论文为不同的语言建模目标设计了四类完形填空任务预训练，一个训练batch中，1/3样例用于bidirectional语言目标，1/3用于Seq2Seq语言目标，1/6分别用于left-to-right和right-to-left的语言目标，联合四种类型的语言目标作为整体的训练目标。

## 4\. BART

Facebook提出的BART模型看做是BERT与GPT结合的降噪自编码器，它是由双向编码器（Bidirectional Encoder）和自回归解码器（Autoregressive Decoder）构成的Sequence-to-Sequence预训练模型，适用于非常广泛的下游任务。

![](https://pic4.zhimg.com/v2-9f70db22f5969d09d487eb2313149f11_1440w.jpg)

BART模型结构图

**4.1 Pre-training BART**

BART预训练时，首先对输入文本/文档通过噪声函数进行转换破坏，然后利用模型对破坏的文本进行复原，并使用解码器输出与原始输入构建交叉熵损失优化模型。BART实验中使用如下5类转化方法破坏输入文本，为模型的输入引入噪声。

![](https://pic3.zhimg.com/v2-160af0ea87779f1cb836c46a00f97b02_1440w.jpg)

文本转换加噪的方法

-   **\[Token Masking\]:** 和BERT相似，随机mask文本中一部分token，每个token使用\[MASK\]进行替换，利用模型预测mask部分的token。
-   **\[Token Deletion\]:** 文本中随机选择token进行删除，利用模型对输入进行重建，预测删除的token。
-   **\[Text Infilling\]**: 采样文本片段span，span长度从泊松分布( $\lambda=3$ )中得到。每个span使用一个\[MASK\]进行替换，若span长度为零，就在文本插入\[MASK\]。训练时利用模型对span中的token进行预测。
-   **\[Sentence Permutation\]:** 文本通过句号被分成句子，这些句子按随机顺序打乱，利用模型将乱序的输入恢复为原始文本的句子顺序。
-   **\[Document Rotation\]:** 随机选择一个token，对文本进行旋转并以该token为开始，利用模型对旋转过的输入进行复原，恢复为原始的文本。

**4.2 Fine-tuning BART**

预训练后的BART可以用于多种类型的下游任务，通过微调后适配使用。

![](https://pica.zhimg.com/v2-3a7c158dbeb4eb49f51efee18d69ec44_1440w.jpg)

BART微调示意图

-   **\[Sequence Classification Tasks\]:** 对于序列分类任务，将文本输入BART模型，将最后一个解码的token的隐层状态向量传入一个多分类器进行类别预测，这种方法和使用BERT的\[CLS\]表示向量进行分类相似。
-   **\[Token Classification Tasks\]:** 对于token分类任务，将文本输入BART模型，将解码器最顶层的各个隐状态向量作为每个token的表示，进行token级别的序列标注或分类。
-   **\[Sequence Generation Tasks\]:** BART是具有自回归解码器的Seq2Seq结构的模型，可以直接应用于序列或文本生成的任务，编码器输入文本，解码器自回归的生成文本。
-   **\[Machine Translation\]:** 对于机器翻译任务，论文使用一个新的随机初始化的编码器代替BART的embedding层，将整个BART堆叠在新的编码器之上，作为机器翻译任务的解码器。

## 5\. T5

谷歌提出了一个统一预训练模型和框架 **T**ext-**t**o-**T**ext **T**ransfer **T**ransformer（**T5**），T5将每个文本处理问题都看成“Text-to-Text”问题，即将文本作为输入，生成新的文本作为输出。通过这种方式可以将不同的 NLP 任务统一在一个模型框架之下，充分进行迁移学习。也就说可以用同样的模型，同样的损失函数，同样的训练过程，同样的解码过程来完成所有 NLP 任务。

![](https://pic3.zhimg.com/v2-bec1e52e9fa3888d62955022eba3340a_1440w.jpg)

T5模型架构示意图：T5将不同的NLP任务都转化成“Text-to-Text”的形式进行建模

**5.1 Input and Output Format**

T5 旨在将所有任务转化成“Text-to-Text”形式，即提供一些输入如文本作为上下文或条件给模型，模型生成输出文本。为了告知模型需要执行的任务类型，在输入的文本前添加任务特定的文本前缀 (task-specific prefifix ) 进行提示，这也就是最早的 Prompt。

```text
“[Task-specific prefix]:[Input text]” -> “[output text]” 
```

比如以下例子：

-   **\[英语翻译成德语\]**

![](https://picx.zhimg.com/v2-5d0d7588a3680f15534de62e88616c4f_1440w.jpg)

-   **\[文本情感分类\]**

![](https://pica.zhimg.com/v2-c8e8bcf98d20e81e657c0321832c2572_1440w.jpg)

**5.2 Model Architectures**

论文对各种Transformer架构变体进行评估对比，确定T5所采用的的模型架构。T5 采用Encoder-Decoder Transformer结构，除了移除 Layer Norm bias、将 Layermalization 放在残差连接之外、使用了[不同的位置Embedding方案](https://zhuanlan.zhihu.com/p/444438914)外，其他和原始的 Transformer 模型几乎一致。

不同架构的一个主要区别因素是模型中不同注意机制使用的 mask 掩码。

-   **Fully-visible mask：**允许每个输出与全部的输入进行注意力计算，和BERT中的mask一致
-   **Causal mask：**禁止输出元素与其未来的输出元素交互，和GPT模型中的mask一致
-   **Causal mask with prefix：**允许在输入序列的一部分上使用Fully-visible mask，其余部分是Causal mask

![](https://pic2.zhimg.com/v2-c1e0c02ae5b43dbfe4c1078dbffaf033_1440w.jpg)

不同的attention mask形式：mask矩阵中黑色的cell表示对应的两个元素可以进行注意力计算，白色的单元表示对应元素不可记性注意力计算。

论文中对预训练模型中的不同Transformer架构和变体进行对比，下图为是三种Transformer架构变体：

-   **Encoder-Decoder**：类似于BART的Seq2Seq结构的模型，编码器是BERT-style的双向语言模型，使用Fully-visible mask，解码器是GPT-style的自回归语言模型，使用Causal mask。
-   **Language Model**：自回归的语言模型，每个输出只能与它左边的信息进行交互，注意力计算使用Causal mask，可参考GPT模型。
-   **Prefix Language Model**：注意力计算使用的是Causal mask with prefix，和UniLM中Seq2Seq LM相似。对于文本生成任务，Prefix LM的输入是source和target文本的拼接，source部分使用Fully-visible mask，而target输出部分使用Causal mask。

![](https://pic3.zhimg.com/v2-55cc526736fcf5bbb7a4c94f6773412a_1440w.jpg)

预训练模型中三类Transformer结构变体：左侧为Encoder-Decoder模型，中间为自回归语言模型，右侧为带前缀语言模型

论文对各个架构变体进行实验对比，表明具有降噪目标的Encoder-Decoder架构性能最好，Encoder和Decoder共享参数的性能与不共享的性能接近，并且共享参数的Encoder-Decoder模型要好于只有Decoder的LM模型和Prefix LM模型。因此**T5选择Encoder-Decoder的Transformer结构**。

![](https://picx.zhimg.com/v2-6efb52f7c2cb00aa76654164d93dc613_1440w.jpg)

不同结构变体的性能对比，P表示12层基础Transformer层堆栈中的参数数量，M表示使用encoder-decoder模型处理序列所需的 FLOPs

**5.3** **Unsupervised Objectives**

论文对模型的预训练目标和方式进行大量的探索和实验，对常见的预训练方式进行修改以适应T5的text-to-text的编解码框架。作者从如下流程图的四个方面进行实验确定适合T5的无监督预训练目标。

\* 作者最终选择如下预训练目标：**BERT-style的预训练方式、Replace spans的文本破坏策略、15%的文本破坏率、长度为3的破坏的span长度。**

![](https://pic3.zhimg.com/v2-73f51512c8137fbdd2d9d1d0f6057c92_1440w.jpg)

论文探索无监督预训练目标的流程图

-   **Disparate High-Level Approaches**

作者选择Prefix LM、BERT-style、 Deshuffling三种自监督的预训练方式进行实验对比，发现**BERT-style的预训练目标性能表现最好，**尽管Prefix LM在翻译任务上超过BERT-style。

![](https://pic2.zhimg.com/v2-4c0f7e2de1313f1d5e1896ba36a9bfad_1440w.png)

三种预训练目标性能对比

-   **Simplifying the BERT Objective**

在确定了BERT-style的预训练目标基础上，作者探索对BERT-style的降噪预训练目标的修改，使其在T5的Text-to-Text任务设置中性能更好并且更高效。通过实验发现各个预训练目标变体之间性能相似，但**Replace corrupted spans不需要解码器预测完整的原始输入序列，因此生成输出序列更短，训练更快速。**

![](https://pic2.zhimg.com/v2-08a0fabd4dda27cfa7bd45180d1a8fd3_1440w.jpg)

一些常见预训练目标的输入和输出例子。&lt;M&gt; 表示共享mask token，&lt;X&gt;、&lt;Y&gt; 和 &lt;Z&gt; 表示分配了不同 ID 的哨兵token。

![](https://pic4.zhimg.com/v2-1d71782f2831d2e005523dec78814d21_1440w.jpg)

BERT-style训练目标的变体性能对比

-   **Varying the Corruption Rate**

作者探索了不同的文本破坏率对模型性能的影响，实验结果发现破坏率对模型性能的影响非常有限，除了在GLUE和SQuAD上50%破坏率性能有明显的下降。更大的破坏率使得模型生成的目标输出更长，可能影响训练效率，因此**作者选择15%的corruption rate**。

![](https://picx.zhimg.com/v2-bf59efdce96ee2fa74e9c59e34fee8e3_1440w.jpg)

-   **Corrupting Spans**

类似于[SpanBERT模型](https://link.zhihu.com/?target=https%3A//aclanthology.org/2020.tacl-1.5.pdf)，作者对比了破坏的span的长度对性能的影响，实验表明**span长度为3时在非翻译任务上表现最好**。

![](https://pic3.zhimg.com/v2-59be2a71c8d1cb0010683fb2c4ceb72a_1440w.jpg)
