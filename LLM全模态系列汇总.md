---
created: 2025-09-01T17:31:30 (UTC +08:00)
tags: [LLM,多模态大模型,MLLM]
source: https://zhuanlan.zhihu.com/p/19157456395
author: 关于作者akaihaoshuai喜欢就去做，不喜欢的才需要理由迷途小书僮、郭达森、程勇也关注了他回答8文章94关注者3,010关注发私信
---

## [Awesome-Multimodal-Next-Token-Prediction](https://link.zhihu.com/?target=https%3A//github.com/LMM101/Awesome-Multimodal-Next-Token-Prediction)（多模态综述）

> [Next Token Prediction Towards Multimodal Intelligence: A Comprehensive Survey](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2412.18619)

![](https://picx.zhimg.com/v2-5a0d2300ed5c46cf8980407a33135633_1440w.jpg)

论文对基于Next Token Prediction（NTP）的多模态学习进行了全面综述。论文首先介绍了多模态学习的背景和NTP在自然语言处理中的成功应用，然后提出了一个统一的多模态学习框架，将多模态信息编码为tokens，并通过[Transformer模型](https://zhida.zhihu.com/search?content_id=252854041&content_type=Article&match_order=1&q=Transformer%E6%A8%A1%E5%9E%8B&zhida_source=entity)进行处理。论文详细介绍了多模态token化、模型架构、统一任务表示、数据集与评估以及面临的挑战。通过分析不同模型的结构和性能，论文为未来多模态智能的研究提供了有价值的参考和方向。

![](https://pic4.zhimg.com/v2-0749a040cb316f8a6713bbc3b062beed_1440w.jpg)

![](https://pic4.zhimg.com/v2-526367847f380129fe2b3a7b3264cb6f_1440w.jpg)

## **[Autoregressive-Models-in-Vision-Survey](https://link.zhihu.com/?target=https%3A//github.com/ChaofanTao/Autoregressive-Models-in-Vision-Survey)（综述）**

> [Autoregressive Models in Vision: A Survey](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2411.05902)
> 
> [https://github.com/ChaofanTao/Autoregressive-Models-in-Vision-Survey](https://link.zhihu.com/?target=https%3A//github.com/ChaofanTao/Autoregressive-Models-in-Vision-Survey)

## [VITA](https://link.zhihu.com/?target=https%3A//github.com/VITA-MLLM/VITA)（实时交互）

> [VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2501.01957)
> 
> [VITA: Towards Open-Source Interactive Omni Multimodal LLM](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2408.05211)

论文介绍了一个名为VITA-1.5的多模态大语言模型，旨在通过三阶段训练策略实现视觉、语言和语音模态的高效融合，以支持实时的多模态交互。

![](https://pic2.zhimg.com/v2-b931ee5bd6bbd2203cfe4e5062bf3469_1440w.jpg)

Vita同时运行了两个模型，即生成模型和监视模型，以支持全双工通信。当生成模型正在生成系统响应时，监视模型会监视环境并检测到有效的用户中断后，它将结合上下文并提供对新用户查询的响应，而生成模型暂停并切换到监视角色。

**架构设计**

-   _**Vision &** Video_

模型采用[InternViT-300M](https://zhida.zhihu.com/search?content_id=252854041&content_type=Article&match_order=1&q=InternViT-300M&zhida_source=entity)作为视觉编码器。输入图像448×448，每张图生成256个token。对于高分辨率图像，采用动态修补策略来捕捉局部细节（一张大图生成多个448×448子图），提高图像理解的准确性。

> 《[How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2404.16821) 》动态修补：根据输入图像的长宽比和分辨率将图像分成 448×448 像素的 1 到 40 个图块，支持高达 4K 分辨率的输入。

如果视频长度短于 4 秒，则均匀采样 4 帧；对于 4 到 16 秒之间的视频，每秒采样 1 帧；对于超过 16 秒的视频，均匀采样 16 帧。不对视频帧应用动态修补，以避免过多的视觉标记影响处理效率。

Vision Adapter：将图像和视频转成token后，通过两层MLP映射对齐。

-   **_Audio_**

由多个下采样卷积层（4 倍下采样）和 24 个 Transformer 块（隐藏大小为 1024）组成。下采样层有助于降低音频特征的帧率，从而提高 LLM 的处理速度。音频编码器具有约 350M 个参数，输出帧率为 12.5Hz。Mel 滤波器组特征用作音频编码器的输入，窗口大小为 25ms，移位为 10ms 。

Audio Adapter：它由多个卷积层和 2 倍下采样组成。

Audio Decoder：

-   1）非自回归（NAR）语音解码器，全局处理文本标记并对语义特征进行建模，目的是生成语音标记的初始分布
-   2）自回归（AR）语音解码器基于NAR解码器产生的语音信息，逐步生成更高质量的语音标记。

然后使用Codec模型的语音解码器将最终的语音标记序列解码为连续的语音信号流（波形）。我们对NAR和AR语音解码器都采用4个LLaMA解码层，其中隐藏大小为896，参数大小约为120M。

**训练**

需要分别训练视觉输入、音频输入和音频输出。

![](https://pic2.zhimg.com/v2-483eb9e2c9c9a46c948420be404bca2b_1440w.jpg)

-   **第一阶段：视觉-语言训练**：

-   **1.1 视觉对齐**：使用20%的描述性标题数据训练视觉适配器，使LLM初步对齐视觉模态。
-   **1.2 视觉理解**：使用全部描述性标题数据，训练视觉编码器、适配器和LLM，使模型能够通过生成自然语言描述来理解图像内容。
-   **1.3 视觉SFT（指令微调）**：使用全部问答数据和20%的描述性标题数据，训练模型以增强其视觉问答能力。

-   **第二阶段：音频输入调整**：

-   **2.1 音频对齐**：使用11000小时的语音-转录配对数据，通过CTC损失函数训练语音编码器，使其能够从语音输入中预测转录文本。然后将语音编码器与LLM集成，训练语音适配器，使LLM能够输出语音数据的转录文本。
-   **2.2 音频SFT**：采样4%的标题数据和20%的问答数据，其中约一半的文本问题被替换为语音版本，训练模型以增强其对语音问题的理解能力。

-   **第三阶段：音频输出调整**：

-   **3.1 编解码器训练**：使用3000小时的文本-语音数据训练编解码模型，使其能够将语音映射为离散标记，并将离散标记解码回语音流。
-   **3.2 NAR+AR解码器训练**：使用文本-语音配对数据，将文本输入LLM以获得嵌入向量，然后通过NAR和AR语音解码器预测对应的语音标记，最终解码为语音信号。

![](https://pic4.zhimg.com/v2-b4f8fa503156868848b5df4dfb0449fd_1440w.jpg)

效果

![](https://pic1.zhimg.com/v2-33162272a68ff78f50e4785083f1e53a_1440w.jpg)

![](https://pic1.zhimg.com/v2-3636ec24bfb639033bc5a4f4a9038280_1440w.jpg)

## **[OpenOmni](https://link.zhihu.com/?target=https%3A//github.com/RainBowLuoCS/OpenOmni)**

> [OpenOmni: Large Language Models Pivot Zero-shot Omnimodal Alignment across Language with Real-time Self-Aware Emotional Speech Synthesis](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2501.04561)
> 
> [GitHub - RainBowLuoCS/OpenOmni](https://link.zhihu.com/?target=https%3A//github.com/RainBowLuoCS/OpenOmni)

论文提出了一种名为 **OpenOmni** 的新型全模态大语言模型，通过利用语言作为支点，实现了图像、文本和语音之间的零样本对齐，并通过轻量级的流式语音解码器实现了实时情感语音生成。

![](https://pic1.zhimg.com/v2-cf95de8978ca2323ef1a090dba603f48_1440w.jpg)

**训练**

由于image-text-speech数据的稀少，同样需要将其拆分成image-text训练和text-speech训练、音频输出训练。

![](https://picx.zhimg.com/v2-f47743363e04f443287abe1bdd55906f_1440w.jpg)

为了使OpenOmni具有增强交互式体验的实时语音生成能力，采用流式语音解码器进行实时的语音生成。

> 论文中提到：MOE层旨在稳定训练和快速收敛。没有这一层，几乎无法成功训练语音解码器。

直接情感偏好优化（DEPO）算法。构建了一个多转向对话偏好数据集，以实现自我意识，涵盖了九种不同的情绪。（就是使用DPO在情感数据集上训练。。。）

![](https://pic4.zhimg.com/v2-a2a634168ec2db54461623acc1fb156d_1440w.jpg)

训练参数如下

![](https://pic1.zhimg.com/v2-6666c8eaf5e6e6076a22073c5a25caba_1440w.jpg)

LLM选用Qwen2.5-7B-Instruct。Vision Encoder采用CLIP-ViT-L ，speech encode采用Whisper-large-v3。流式speech decoder采用Qwen2.5-0.5B-Instruct 。不同模态之间都采用MLP adapter对齐。对于自回归模型，Speech Tokenizer采用GLM4-Voice的16K词表。对于非自回归模型，Speech Tokenizer采用CosVoice的6K词表。

**效果**

评测上看，比VITA还要好一些。

![](https://picx.zhimg.com/v2-03fd5b114fc256aa88d524c67db8cda7_1440w.jpg)

![](https://pic3.zhimg.com/v2-6a97ac34a47f0c66153cc1c3c2ddf820_1440w.jpg)

## [MinMo](https://link.zhihu.com/?target=https%3A//funaudiollm.github.io/minmo/)（实时交互）

> [MinMo: A Multimodal Large Language Model for Seamless Voice Interaction](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2501.06282v1)

这篇论文介绍了一个名为MinMo的多模态大语言模型，旨在实现无缝语音交互。无缝交互的核心是存在另外一个系统（Full Duplex Predictor）用于根据输入音频判断是否重新响应生成还是继续输出。

-   提出了一种新颖的语音解码器，能够将文本LLM的输出高效地转换为语音。
-   支持用户随时打断系统并进行对话。
-   通过指令跟随能力，MinMo能够根据用户指令生成具有特定情感、语速和方言的语音。

![](https://pic2.zhimg.com/v2-99251c85a5f4f0ba8eb32fedc05ecdf1_1440w.jpg)

模型训练的效果极佳

![](https://pic3.zhimg.com/v2-a4ee6e716a6d40acc115e3059ca2f5d6_1440w.jpg)

**模型**

MinMo基于一个预训练的文本LLM（Qwen2.5-7B-instruct），通过轻量级的模态对齐方法将其扩展为多模态模型。其架构包括以下几个关键模块：

-   **Voice Encoder**：使用预训练的SenseVoice-large模块，支持多语言语音识别、情感识别和音频事件检测。
-   **Input Projector**：包含两层Transformer和一层CNN，用于多模态维度对齐和下采样。
-   **Output Projector**：单层线性模块，用于将LLM的输出与语音解码器对齐。
-   **Voice Token LM**：基于预训练的CosyVoice 2 LM模块，自回归地生成语音令牌。
-   **Full Duplex Predictor**：单层Transformer和线性softmax输出层，用于实时预测是否响应用户指令或暂停系统输出。
-   **Token2wav Synthesizer**：将语音令牌转换为波形，包含预训练的流匹配模型和声码器。

![](https://picx.zhimg.com/v2-a7da5ee89234009182f177c288cec36f_1440w.jpg)

**训练**

MINMO的训练任务包括四个类别，包括_语音到文本_，_文本到语音_，_语音到语音_以及_语音到连续的_任务。

![](https://picx.zhimg.com/v2-8b2092f87b71dd396fd2e0260c96b6e7_1440w.jpg)

语音到文本任务。此类别包括大约120万小时的语音文本配对数据，包括诸如自动语音识别（ASR），语音到文本翻译（S2TT），语言识别（LID），上下文偏见语音识别，语音情绪识别等任务（SER），音频事件检测（AED），扬声器分析，口语平滑。这些任务的培训数据以CHATML格式组织，由以下示例说明：

![](https://pica.zhimg.com/v2-4c25f6b45e877d13e5cc375024e8029c_1440w.jpg)

文本到语音任务。该类别的数据主要由基本语音综合数据组成，该数据与用于培训Cosyvoice 2的数据相同。它包括170,000小时的文本语音配对数据并支持四种语言：中文，英语，韩语和日语。此外，大约有1,000小时的音频生成数据由说明控制。

![](https://pic1.zhimg.com/v2-ea47ebf4b510571a6a5015d15943293a_1440w.jpg)

语音到语音任务。语音到语音的数据主要是通过模拟来源的，其中包括大约10,000小时的多转换语音和100个小时的样式控制的多转交谈语音。

-   利用Cosyvoice的zero-shot中文本生成方法将用户文本转换为用户语音。将Cosyvoice的基本模型与选定扬声器的2小时数据进行微调，以创建针对目标扬声器的语音合成模型，该模型称为Cosyvoice-SFT。使用zero-shot生成来进行用户语音综合的优点在于它的能力确保生成的用户语音中的多样性，从而增强Minmo的普遍性。
-   从ASR数据中选择合适的真实语音作为用户语音查询，并使用相应的文本作为QWEN-MAX的输入来生成响应文本，然后使用Cosyvoice将其合成为助手语音-SFT模型。这种方法进一步增强了模型对真实用户音频输入的鲁棒性。

语音到控制任务。语音到控制的数据主要由两个部分组成。第一部分是从现有的_真实_语音交互数据中提取的，而第二部分是使用文本对话数据_模拟的_。用于训练全双工交互能力，数据量为4000小时。

MinMo通过四个阶段的对齐训练：

1.  **语音到文本对齐**：使用语音到文本数据对输入投影器和语音编码器进行训练，同时通过LoRA更新LLM。
2.  **文本到语音对齐**：使用文本到语音数据训练输出投影器和语音令牌语言模型。
3.  **语音到语音对齐**：继续使用语音到语音数据训练输出投影器和语音令牌语言模型。
4.  **全双工交互对齐**：训练全双工预测器模块，使其能够根据LLM的语义理解能力决定是否响应用户指令。

## [Baichuan-Omni-1.5](https://link.zhihu.com/?target=https%3A//github.com/baichuan-inc/Baichuan-Omni-1.5)

> [Baichuan-Omni-1.5 Technical Report](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2501.15368)
> 
> [https://github.com/baichuan-inc/Baichuan-Omni-1.5](https://link.zhihu.com/?target=https%3A//github.com/baichuan-inc/Baichuan-Omni-1.5)

Baichuan-Omni-1.5，这是一个强大的多模态模型，能够处理文本、图像、音频和视频等多种模态。

![](https://picx.zhimg.com/v2-7a94becf4d8f0578e557fa3536c197ed_1440w.jpg)

论文通过以下三个关键方面来解决多模态融合的问题：

**1\. 高质量多模态预训练数据**

-   **数据来源**：构建了包含文本、图像-文本、视频-文本、音频-文本及其交互的全面多模态数据集。数据来源包括公开数据集、内部数据以及合成数据。
-   **数据处理**：

-   **图像数据**：包括交错图像-文本数据、字幕数据和问答数据。通过合成技术生成高质量的中文字幕和交错数据，以增强模型对中文的理解能力。
-   **视频数据**：涵盖视频分类、动作识别和时间定位任务的数据。使用 GPT-4o 生成高质量的视频字幕。
-   **音频数据**：分为音频理解数据（如语音识别、音频问答）和音频生成数据（如文本到语音）。通过交错文本和音频数据来提升模型对复杂上下文的建模能力。
-   **跨模态交互数据**：通过将文本中的部分句子替换为音频元素，生成图像-音频-文本和视频-音频-文本交互数据，以增强跨模态交互能力。

**2\. 模型架构**

-   **视觉分支**：使用 NaViT（Qwen2-VL 的视觉编码器）处理图像和视频输入，将其转换为视觉 tokens，然后通过视觉投影器压缩特征。
-   **音频分支**：引入 **Baichuan-Audio-Tokenizer**，基于残差向量量化（RVQ）和多目标训练，将音频信号转换为离散 tokens，并通过音频解码器将 tokens 解码为语音波形。
-   **多模态融合**：视觉和音频 tokens 与文本 tokens 一起输入到预训练的大语言模型（LLM）中，实现多模态信息的融合。

**3\. 多阶段多模态训练策略**

-   **图像-文本预训练**：通过图像-文本数据对视觉编码器和 LLM 进行对齐训练，分为两个阶段：

-   **阶段 I**：仅训练视觉投影器。
-   **阶段 II**：解冻视觉编码器和 LLM，进一步提升图像和文本的对齐能力。

-   **图像-音频-文本预训练**：在图像-文本预训练的基础上，引入音频数据，通过交错音频和文本数据进行预训练，分为两个阶段：

-   **阶段 I**：仅训练音频嵌入层和音频头。
-   **阶段 II**：解冻除视觉编码器和音频 tokenizer 外的所有参数。

-   **多模态预训练**：使用高质量的跨模态交互数据（图像-音频-文本和视频-音频-文本）进行训练，支持长音频和视频流。
-   **多模态监督微调**：通过监督微调进一步提升模型在多模态任务上的表现，使用包含多种模态（文本、音频、图像、视频）的指令数据进行训练。

![](https://pic2.zhimg.com/v2-bdc999f2ee143c53573a5c09646be57f_1440w.jpg)

## [LlamaFusion](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2412.15188)（LLM->MLLM时保证质量不下降）

> [LlamaFusion: Adapting Pretrained Language Models for Multimodal Generation](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2412.15188)

目前大多数多模态生成模型都是从头开始训练，这需要大量的计算资源，且在语言任务上的性能往往不如预训练的文本语言模型。

**[LMFusion](https://zhida.zhihu.com/search?content_id=252854041&content_type=Article&match_order=1&q=LMFusion&zhida_source=entity)**，一个将预训练的文本语言模型（如Llama-3）扩展为多模态生成模型的框架。通过引入模态特定的Transformer模块和冻结文本模块，LMFusion在保留强大语言能力的同时，显著提升了图像理解和生成能力。

![](https://pic3.zhimg.com/v2-caf2f7375f93ff9b1961669ef4c2bf4c_1440w.jpg)

**模型架构**

-   **基础架构**：LMFusion是一个基于Transformer的解码器模型，包含N层Transformer层。它在预训练的Llama-3模型基础上，引入了额外的图像专用Transformer模块。
-   **模态特定模块**：引入模态特定的前馈网络（FFN）和查询-键-值（QKV）投影，分别处理文本和图像数据。文本模块继承自Llama-3并冻结权重，以保留语言能力；图像模块则用于视觉理解和生成任务。
-   **跨模态交互**：在自注意力层中，文本和图像的表示可以跨模态边界相互关注，允许跨模态信息融合。
-   **输入与输出**：

-   文本输入通过线性嵌入层投影到文本隐藏状态。
-   图像输入通过U-Net降采样器投影到图像表示。
-   输出时，文本隐藏状态通过语言模型头投影到文本logits，图像隐藏状态通过U-Net上采样器预测图像噪声。

**训练策略**

-   **冻结文本模块**：冻结所有文本模块（包括投影层、QKV、FFN和输出层），仅训练图像模块，避免对语言能力的干扰。
-   **独立学习率**：为文本和图像模块设置独立的学习率，冻结文本模块时，其学习率为0，图像模块使用单独的学习率进行训练。

## [VILA](https://link.zhihu.com/?target=https%3A//github.com/NVlabs/VILA)

> [NVILA: Efficient Frontier Visual Language Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2412.04468)
> 
> [https://github.com/NVlabs/VILA](https://link.zhihu.com/?target=https%3A//github.com/NVlabs/VILA)

[VILA](https://link.zhihu.com/?target=https%3A//github.com/NVlabs/VILA)旨在优化模型的效率和准确性。通过采用“扩展-压缩”策略，NVILA能够在处理高分辨率图像和长视频时保持高效性，同时在多个图像和视频基准测试中取得了与现有最先进模型相当甚至更好的性能。

![](https://pic4.zhimg.com/v2-1eff311bad3dada1aabc02bb928cae51_1440w.jpg)

**高效模型架构（Efficient Model Architecture）**

-   **空间“扩展-压缩”（Spatial “Scale-Then-Compress”）**：

-   **扩展**：通过Dynamic-S2技术，将输入图像按比例放大到更高分辨率（如896×896），并将其分割成多个小块（tiles），分别处理后再拼接。这种方法可以保留更多图像细节，提升模型在文本密集型任务上的性能。
-   **压缩**：采用空间池化（spatial pooling）技术，将图像块中的token数量减少。例如，通过3×3的空间池化，将每个图像块的token数量从256减少到121。虽然这会导致一定的性能损失，但通过额外的视觉编码器预训练（VEP）阶段，可以恢复大部分性能。

-   **时间“扩展-压缩”（Temporal “Scale-Then-Compress”）**：

-   **扩展**：增加视频输入的采样帧数，从8帧扩展到32帧甚至更多，以提高模型对长视频的理解能力。
-   **压缩**：采用时间池化（temporal pooling）技术，通过平均相邻帧的特征来减少token数量。例如，将32帧视频的时间池化因子设为4，从而将token数量减少到原来的1/4。这种方法可以有效减少冗余信息，同时保留重要的时空特征。

![](https://pica.zhimg.com/v2-a18fc9a9d793d5b77b015298916d4e9c_1440w.jpg)

**高效训练（Efficient Training）**

-   **数据集修剪（Dataset Pruning）**：

-   采用DeltaLoss方法对训练数据进行筛选，去除那些对模型训练贡献较小的样本（如过于简单或过于复杂的样本）。通过这种方式，可以在不显著降低模型性能的情况下，减少训练数据量，从而加速训练过程。

![](https://pic4.zhimg.com/v2-26f3f4189d8fc70f77b6e2938153ffdb_1440w.jpg)

-   **FP8训练（FP8 Training）**：

-   利用NVIDIA Hopper架构支持的FP8精度进行训练。与传统的BF16精度相比，FP8可以在保持模型精度的同时，显著提高训练速度和内存效率。论文中提到，在不使用梯度检查点（gradient checkpointing）的情况下，FP8训练可以将训练速度提高2倍；即使在使用梯度检查点的情况下，FP8训练仍可提供1.2倍的速度提升。

**高效微调（Efficient Fine-Tuning）**

-   在微调阶段，论文发现对于视觉编码器（ViT）和语言模型（LLM）的微调需要分别设置不同的学习率，其中ViT的学习率通常比LLM的学习率小5-50倍。此外，论文还提出了一种基于LayerNorm的微调方法，该方法在计算效率上优于LoRA，同时能够保持与LoRA相当的性能。

**高效部署（Efficient Deployment）**

-   在模型部署阶段，论文开发了一种专用的推理引擎，采用量化技术（如W8A8和W4A16）来加速模型的推理过程。具体来说，在预填充阶段（prefilling stage），对视觉塔（vision tower）进行W8A8量化，以减少首次输出延迟（Time-To-First-Token, TTFT）；在解码阶段（decoding stage），对LLM主干进行W4A16量化，以提高解码吞吐量。

## **[Awaker](https://link.zhihu.com/?target=https%3A//github.com/MetabrainAGI/Awaker)（MOE优化多任务MLLM）**

> [Awaker2.5-VL: Stably Scaling MLLMs with Parameter-Efficient Mixture of Experts](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2411.10669)
> 
> [https://github.com/MetabrainAGI/Awaker](https://link.zhihu.com/?target=https%3A//github.com/MetabrainAGI/Awaker)

本文提出了Awaker2.5-VL，这是一个基于混合专家（MoE）架构的多模态大语言模型，旨在解决多任务冲突问题。通过设计多个任务特定的专家模型和一个全局专家，并采用低秩适配（LoRA）结构来加速训练和推理，Awaker2.5-VL在多个最新的多模态基准测试中取得了优异的性能。

![](https://pic3.zhimg.com/v2-46b9300c2d54f56d733e14cc7b9c7ca0_1440w.jpg)

Awaker2.5-VL的训练分为三个阶段：

1.  **初始化训练（Stage I）**：在基础模型上添加LoRA模块进行训练，冻结基础模型，仅训练LoRA参数。
2.  **MoE训练（Stage II）**：用MoE模块替换第一阶段的LoRA模块，初始化MoE模块的专家参数为第一阶段训练的LoRA参数，冻结基础模型，仅训练MoE模块（包括门控层和所有专家）。
3.  **指令微调（Stage III）**：冻结MoE模块的门控层，仅训练专家。

![](https://picx.zhimg.com/v2-ca49aff5ef98b1f6fcba29e8cd5bc2bb_1440w.jpg)

构建了一个包含约1200万条数据的数据集，其中英语数据700万条，中文数据500万条。数据来源包括开源数据集（如Cambrain、LLaVAOneVision等）和自建的中文指令数据集。

![](https://pic2.zhimg.com/v2-eae9bbb80e924b529cea75c361f77ae5_1440w.jpg)

效果评测

## [DeepSeek-VL2](https://link.zhihu.com/?target=https%3A//github.com/deepseek-ai/DeepSeek-VL2)

> [DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2412.10302)
> 
> [https://github.com/deepseek-ai/DeepSeek-VL2](https://link.zhihu.com/?target=https%3A//github.com/deepseek-ai/DeepSeek-VL2)

论文介绍了DeepSeek-VL2，这是一个基于混合专家（Mixture-of-Experts, MoE）架构的视觉-语言模型系列，旨在显著提升其前身DeepSeek-VL的性能和效率。

![](https://pic1.zhimg.com/v2-0eb1fb66f1abd0a310bed3df6b584c84_1440w.jpg)

主要贡献包括：

1.  **动态图像分割策略**：通过将高分辨率图像分割为多个小块，动态处理不同宽高比的图像，提高了视觉理解能力。
2.  **优化的语言模型架构**：采用DeepSeekMoE语言模型和多头潜注意力机制（MLA），显著提高了训练和推理效率。
3.  **改进的视觉-语言数据集**：通过增强数据的质量、数量和多样性，提升了模型在多种任务上的表现。
4.  **多模态性能提升**：在多个基准测试中，DeepSeek-VL2展现了与现有开源模型相当或更优的性能，并且在视觉定位等任务上表现出色。 论文还展示了DeepSeek-VL2在视觉问答、图像描述、图表理解、视觉故事创作等任务上的能力，并提出了未来的研究方向。

## **[LLaVA-Mini](https://link.zhihu.com/?target=https%3A//github.com/ictnlp/LLaVA-Mini)（视觉token加速）**

> [LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2501.03895)

论文试图解决**大规模多模态模型（LMMs）在处理视觉信息时的计算效率问题**。例如，使用CLIP ViT-L/336px编码器时，一张图像会被编码为24×24=576个视觉token，这使得模型在处理高分辨率图像或视频时效率低下。

**原理**

先分析LMM中视觉令牌的重要性。将attention score显示出来。

![](https://pica.zhimg.com/v2-c1cbd984c434518d3bf6265cbc6303a6_1440w.jpg)

分配给视觉令牌的注意力在各个层之间差异很大。视觉令牌在较早的层中受到更多关注，但是这种注意力在更深层次的层中急剧下降，超过80％的注意力集中在指导令牌上。

![](https://pica.zhimg.com/v2-19757b48bba26e0875ac2be211204d5c_1440w.jpg)

计算了每一层注意分布的熵。发现在早期层中，对视觉令牌的注意力要高得多，这表明大多数视觉令牌在早期层中都均匀地关注。

将Attention可视化后可以看出，所有视觉令牌在早期层中都是至关重要的，并且不可避免地减少其数量会导致视觉信息的丧失。

![](https://pica.zhimg.com/v2-36e7e7209a438149e655945459b1ff84_1440w.jpg)

为了证实上述发现，评估了当视觉令牌在不同层上删除时LMM的视觉理解能力。

![](https://pic3.zhimg.com/v2-ee228f5bf60e0cfc18876b1a64e6a8f8_1440w.jpg)

在早期层中删除视觉令牌会导致视觉理解能力的完全丧失，而在较高层中删除令牌的效果很小，该模型保留了其许多原始性能。

**模型**

![](https://pic2.zhimg.com/v2-157b83b831c3efb7cce651d8f1ec9bfb_1440w.jpg)

**LLaVA-Mini模型**：

-   **模态预融合（Modality Pre-fusion）**：基于上述分析，论文提出在将视觉token输入LLM之前，先通过一个预融合模块将视觉信息融合到文本token中。这样可以在减少视觉token数量的同时，保留更多的视觉信息。
-   **视觉token压缩（Vision Token Compression）**：论文引入了一个基于查询的压缩模块，将大量的视觉token压缩为极少数的token（例如1个token）。压缩模块通过交叉注意力机制选择性地提取重要的视觉信息，并通过二维正弦位置编码保留图像的空间信息。
-   **处理高分辨率图像和视频**：对于高分辨率图像，论文通过将图像分割为多个子图像并分别处理，然后将子图像的视觉token压缩并融合。对于视频，论文通过逐帧处理并压缩每帧的视觉token，从而支持长视频的高效处理。

## [PyramidDrop](https://link.zhihu.com/?target=https%3A//github.com/Cooperx521/PyramidDrop)（token裁剪加速）

> [PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2410.17247)
> 
> [https://github.com/Cooperx521/PyramidDrop](https://link.zhihu.com/?target=https%3A//github.com/Cooperx521/PyramidDrop)

以前的方法试图减少 LVLM 早期层之前或内部的图像标记数量。然而，这些策略不可避免地会导致关键图像信息的丢失，最终降低模型性能。本文进行了实证研究，表明浅层中的所有视觉标记对于 LVLM 都是必需的，并且模型较深层的标记冗余度逐渐增加。

提出了 PyramidDrop，这是一种用于 LVLM 的视觉冗余度减少策略，可在性能损失可忽略不计的情况下提高其训练和推理效率。具体来说，将 LVLM 划分为几个阶段，并在每个阶段结束时以预定义的比例删除部分图像标记，从而在模型层之间创建类似金字塔的视觉标记。删除基于轻量级相似度计算，时间开销可忽略不计。

![](https://pic2.zhimg.com/v2-25e229e1c0e1ce5b0148f4f62fb08c53_1440w.jpg)

具体方法如下：

**1、研究视觉token冗余**：

-   通过实验发现，LVLMs在浅层需要所有图像token来理解图像，但随着层数增加，图像token的冗余逐渐增加。
-   在浅层，模型关注大多数图像token以获得全局信息；而在深层，模型逐渐聚焦于与指令相关的少数token。

**2、PyramidDrop策略**：

-   将LVLM分为多个阶段，在每个阶段末尾根据预定义比例丢弃部分图像token，形成金字塔状的视觉token分布。
-   使用轻量级注意力模块对图像token进行排序，计算开销可忽略不计。
-   在浅层保留所有图像token以避免信息丢失，随着层数增加逐步减少token数量以提高效率。

**3、效率分析**：

-   PyramidDrop引入的额外计算开销主要在于图像token排序的相似性计算，计算复杂度为O(n)，且在整个前向传播过程中仅进行S-1次。
-   通过减少冗余token，PyramidDrop能够显著降低模型的计算成本。例如，当压缩比λ=0.5且分为4个阶段时，理论上可节省约53.2%的计算成本。

## **[LLaVA-CoT](https://link.zhihu.com/?target=https%3A//github.com/PKU-YuanGroup/LLaVA-CoT)**

> [LLaVA-o1: Let Vision Language Models Reason Step-by-Step](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2411.10440v1)
> 
> [https://github.com/PKU-YuanGroup/LLaVA-CoT](https://link.zhihu.com/?target=https%3A//github.com/PKU-YuanGroup/LLaVA-CoT)

论文介绍了一种名为 **LLaVA-o1** 的新型视觉语言模型，旨在通过结构化、多阶段的推理过程解决现有 VLMs 在复杂推理任务中的不足。LLaVA-o1 将回答生成过程分解为摘要、描述、推理和结论四个阶段，并通过构建 LLaVA-o1-100k 数据集和阶段级束搜索方法，显著提升了模型在多模态推理任务中的性能。

![](https://pica.zhimg.com/v2-c7770a3477960b229f4b67acd5a8c890_1440w.jpg)

**推理时间扩展的阶段级束搜索**

-   **方法概述**：在推理阶段，LLaVA-o1 利用其结构化输出设计，通过阶段级束搜索（Stage-level Beam Search）进一步提升推理能力。具体步骤如下：

1.  对每个推理阶段生成 N 个候选响应。
2.  随机选择 2 个候选响应，让模型判断哪个更好，并保留更好的响应。
3.  重复上述过程 N-1 次，直到所有阶段处理完毕。

-   **优势**：这种方法在每个推理阶段进行有效的验证，平衡了质量控制和计算效率，显著提高了复杂推理任务的准确性。

![](https://pic4.zhimg.com/v2-9a0e19cbfef3391908b2baecc54fdee7_1440w.jpg)

## 参考文章

[Awesome-Multimodal-Next-Token-Prediction](https://link.zhihu.com/?target=https%3A//github.com/LMM101/Awesome-Multimodal-Next-Token-Prediction)

[Next Token Prediction Towards Multimodal Intelligence: A Comprehensive Survey](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2412.18619)

[Autoregressive Models in Vision: A Survey](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2411.05902)

[https://github.com/ChaofanTao/Autoregressive-Models-in-Vision-Survey](https://link.zhihu.com/?target=https%3A//github.com/ChaofanTao/Autoregressive-Models-in-Vision-Survey)

[VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2501.01957)

[VITA: Towards Open-Source Interactive Omni Multimodal LLM](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2408.05211)

[OpenOmni: Large Language Models Pivot Zero-shot Omnimodal Alignment across Language with Real-time Self-Aware Emotional Speech Synthesis](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2501.04561)

[GitHub - RainBowLuoCS/OpenOmni](https://link.zhihu.com/?target=https%3A//github.com/RainBowLuoCS/OpenOmni)

[MinMo](https://link.zhihu.com/?target=https%3A//funaudiollm.github.io/minmo/)

[MinMo: A Multimodal Large Language Model for Seamless Voice Interaction](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2501.06282v1)

[Baichuan-Omni-1.5 Technical Report](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2501.15368)

[https://github.com/baichuan-inc/Baichuan-Omni-1.5](https://link.zhihu.com/?target=https%3A//github.com/baichuan-inc/Baichuan-Omni-1.5)

[LlamaFusion: Adapting Pretrained Language Models for Multimodal Generation](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2412.15188)

[NVILA: Efficient Frontier Visual Language Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2412.04468)

[https://github.com/NVlabs/VILA](https://link.zhihu.com/?target=https%3A//github.com/NVlabs/VILA)

[Awaker2.5-VL: Stably Scaling MLLMs with Parameter-Efficient Mixture of Experts](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2411.10669)

[https://github.com/MetabrainAGI/Awaker](https://link.zhihu.com/?target=https%3A//github.com/MetabrainAGI/Awaker)

[DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2412.10302)

[https://github.com/deepseek-ai/DeepSeek-VL2](https://link.zhihu.com/?target=https%3A//github.com/deepseek-ai/DeepSeek-VL2)

**[LLaVA-Mini](https://link.zhihu.com/?target=https%3A//github.com/ictnlp/LLaVA-Mini)**

[LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2501.03895)

[GPT-4o System Card](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2410.21276)

[GPT4o Realtime voice功能的复现路径](https://zhuanlan.zhihu.com/p/1387019285)

[PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2410.17247)

[https://github.com/Cooperx521/PyramidDrop](https://link.zhihu.com/?target=https%3A//github.com/Cooperx521/PyramidDrop)

[LLaVA-o1: Let Vision Language Models Reason Step-by-Step](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2411.10440v1)

[https://github.com/PKU-YuanGroup/LLaVA-CoT](https://link.zhihu.com/?target=https%3A//github.com/PKU-YuanGroup/LLaVA-CoT)

[Image Tokenizer与Autoregressive Image Generation](https://zhuanlan.zhihu.com/p/707759472?utm_psn=1813307767139729408)

[训练VLM(视觉语言模型)的经验](https://zhuanlan.zhihu.com/p/890327005)
