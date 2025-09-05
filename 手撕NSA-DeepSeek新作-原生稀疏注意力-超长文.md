---
created: 2025-09-02T14:59:26 (UTC +08:00)
tags: [DeepSeek,LLM]
source: https://zhuanlan.zhihu.com/p/24841366485
author: 关于作者小冬瓜AIGC原创课程➡️ 公众号：手撕LLMJOYWIN、Nasusu、Sam聊算法也关注了他回答9文章39关注者5,274关注他发私信
---
## 1\. [Native Sparse Attention](https://zhida.zhihu.com/search?content_id=253990501&content_type=Article&match_order=1&q=Native+Sparse+Attention&zhida_source=entity)概览

### 1.1 技术概要

DeepSeek开年王炸不断，[R1推理](https://zhida.zhihu.com/search?content_id=253990501&content_type=Article&match_order=1&q=R1%E6%8E%A8%E7%90%86&zhida_source=entity)(Reasoning)训练范式的热度愈演愈烈，随着带来的问题就是inference成本剧增，尤其是在解决难题时需要[LongCoT](https://zhida.zhihu.com/search?content_id=253990501&content_type=Article&match_order=1&q=LongCoT&zhida_source=entity)。而DeepSeek近期发布了`NSA`(Native Sparse Attention)，目的就是进一步减少部署的成本，在开篇提炼关键点：

1.  Native Sparse：指的是做预训练的原生模型就是稀疏的注意力结构, 而不是在预训练好的模型后再进一步稀疏化处理，这里的稀疏化是指KV序列的稀疏化，而不是特征稀疏化。

![](https://pic1.zhimg.com/v2-10138316d23d40b0ddc8b8a4b6fe7774_1440w.jpg)

2\. 我认为`NSA`是由inference特性导向的设计，`NSA`的目的是减轻long-context的注意力计算，动机就是筛选出更少的KV，而另外一个经典的[MLA](https://zhida.zhihu.com/search?content_id=253990501&content_type=Article&match_order=1&q=MLA&zhida_source=entity)实际上是发生在注意力之前，即针对Wk和Wv投影层进行改进，目的是减少[KV-Cache](https://zhida.zhihu.com/search?content_id=253990501&content_type=Article&match_order=1&q=KV-Cache&zhida_source=entity)的，而不是减少注意力的计算。`NSA`和MLA发生在不同层，二者并不冲突。另外常见的[MQA](https://zhida.zhihu.com/search?content_id=253990501&content_type=Article&match_order=1&q=MQA&zhida_source=entity)/GQA我认为也是inference导向的模型改进。

![](https://pic1.zhimg.com/v2-8d55032a8fc8542173b428385c3452d0_1440w.jpg)

3\. `NSA`在KV序列的不同尺度上取舍：压缩、选择和就近滑动窗口，实际上是把控全局信息，把控局部信息和把控就近信息；事实上attention本身就有这种稀疏化建模能力，而`NSA`是将稀疏化能力进一步分层再组合。

![](https://pic4.zhimg.com/v2-3afc35658c74b2b4ed92df8368a8b5cf_1440w.jpg)

### 1.2 问题

既然是原生的Sparse设计，那么提出一些疑问来串联全篇：

1.  为什么要做原生的Sparse设计，而不是在已预训练好的稳定模型上做稀疏化处理？
2.  在文本序列稀疏化处理时，是否会产生信息缺失、不连续和参数不可导？
3.  `NSA`的目的是什么？ 通过什么方式来加速？
4.  `NSA` 的KV-Cache是否是Sparse存储的？
5.  `NSA`的三个Attention各自目的是什么？是如何实现的？
6.  `NSA`是否能减少超大的矩阵计算的空间复杂度问题
7.  `NSA`的training/forward/prefill/decoding各阶段的时空/计算复杂度是怎么样的？
8.  `NSA` 具体来说是seq-block-wise sparse的，那么block-wise的attention用的是MHA/MQA还是GQA？
9.  `NSA` 的内核是如何减少对SRAM的访存的？
10.  如何设计出一种低cache和低SRAM的注意力机制？
11.  为什么压缩注意力和选择注意力要引入stride参数？
12.  `NSA`的KV压缩与`MOBA` 的KV压缩有什么区别？
13.  `NSA`稀疏处理是发生在计算注意力分数前还是后，有什么区别?
14.  抛开`NSA`,为什么说attention本身有稀疏性？长文本建模中attention的稀疏性有什么影响？
15.  `NSA` 为什么比`Flash Attention` 更快？
16.  给定压缩块大小、context长度、选择块大小、滑窗大小和Q头数/KV头数，计算相较标准多头注意力加速多少倍？

## 2\. Inference阶段和瓶颈分析

自回归的LLM模型在inference时分两个阶段：

1.  预填充：根据用户输入prompt，填充每个attention层的KV-Cache，该attention计算过程是并行的；比如数学竞赛题，题目通常非常简短
2.  解码：在预填充后，要逐个token进行解码，该计算过程是串行的；

在R1获得成功后，随着而来的就是在部署时产生了更长的文本生成，其中的LongCoT就是发生在解码阶段。上述两阶段的算力成本是不同的，同等处理token价格，解码阶段显然更贵。

![](https://pic1.zhimg.com/v2-796f7e64110b1e685b99ab44269228fc_1440w.jpg)

我们可以从两个方面来看inference效率瓶颈

1.  空间：在解码过程中，我们可以使用continue-batch来对多个请求解码，而这里的能跑多少batch size对应的KV-Cache管理是关键，MLA是能够急剧的减少KV-Cache存储，从而在推理服务中能跑更大的批次。另外在超长上下文的注意力计算中，可以通过块注意力方法来解决部分存储爆炸问题。
2.  计算：随着上下文长度增长，注意力所需要的计算复杂度是平方复杂度的，如何减少计算量成为关键。

![](https://pic4.zhimg.com/v2-f9d3537824ac824a13b6ce3882fb76a5_1440w.jpg)

对于注意力设计，我们看到有Flash Attention这样优秀的工作，但是并未通过减少注意力计算量而进行加速。假设我们的上下文是无限长，我们可以如下处理，控制上下文序列的长度：

1.  RNN/线性Attention)，将过去的状态压成隐状态
2.  窗口注意力：限制注意力计算的可视范围
3.  稀疏化：按照一定的识别只筛选有限的token做注意力计算
4.  ......

![](https://picx.zhimg.com/v2-52813393c89c831c6fe1e804ad0522dd_1440w.jpg)

从上述三种方法可知，所以我们只要控制好计算的系列长度就能减少计算量。

在上面铺垫这么多后，我们先说明`NSA`的动机：`NSA`是为了减少注意力计算量而生的

**Sparse**：减少计算注意力的context，这里的context有两个层面一种是文本层面，一种是KV层面，`NSA`是后者。

**Native**：我们可以在已有注意力上魔改，为什么还要从原生开始预训练，最关键的还是魔改的稀疏注意力性能退化严重, 总结以下结论是稀疏化处理后20%只能恢复70%的分数，并且稀疏化的注意力不稳定，非常脆弱。

> Performance Degradation: Applying sparsity post-hoc forces models to deviate from their pretrained optimization trajectory. As demonstrated by Chen et al. (2024), top 20% attention can only cover 70% of the total attention scores, rendering structures like retrieval heads in pretrained models vulnerable to pruning during inference

![](https://pic2.zhimg.com/v2-1bc933683122ef7c5d53a7084c511029_1440w.jpg)

另外也提到稀疏注意力训练的难题：参数不可训练和反向传播低效问题。

### 3.1 ReThink 注意力机制

通常注意力计算前，我们把token向量投影成$\textbf{q},\textbf{k},\textbf{v}$, 我们可以用第$t$个词元对应的$\textbf{q}_t$和前$t$个$\textbf{k}_{:t}$ 和$\textbf{v}_{:t}$ 计算注意力

$\mathbf{o}_t=\textbf{Attn}(\mathbf{q}_t, \mathbf{k}_{:t}, \mathbf{v}_{:t})$

其中$\textbf{q}_t, \textbf{o}_t\in \mathrm{R}^{d \times 1}$, $\mathbf{k}_{:t},\mathbf{v}_{:t}\in \mathrm{R}^{d\times t}$

$\textbf{Attn}(\mathbf{q}_t,\mathbf{k}_{:t},\mathbf{v}_{:t})=\sum_{i=1}^t\frac{ \alpha_{t,i}\mathbf{v}_i}{\sum_{j=1}^t \alpha_{t,j}}, \quad \alpha_{t,i}=e^{\frac{\mathbf{q}_t^\top \mathbf{k}_i}{\sqrt{d_k}}}$

在上述公式中，我们可以控制$\textbf{k}_{:t}$ 和$\textbf{v}_{:t}$ 的数量，如$\textbf{k}_{t/2:t}$ 和$\textbf{v}_{t/2:t}$ , 仍然能够得到同个维度的输出$\textbf{o}'_t$, 但是注意力输出是有区别的，如下所示：

![](https://pic3.zhimg.com/v2-4f832bc087066fd72236434f1969a14e_1440w.jpg)

在特征视角下，注意力特征既是序列token表征的组合，那么我们可以知道：

1.  本身注意力分数就能捕捉稀疏性，比如滤掉注意力分数少于一定阈值的$\textbf{q}, \textbf{k}$
2.  但上述的策略里我们要先算完整的注意力分数矩阵，并未在计算注意力分数前进行筛选，导致计算量仍不能降下去。
3.  所以我们需要的是识别哪些$\textbf{k}, \textbf{v}$ 要用于做注意力。

### 3.2 NSA简易理解

`NSA`在KV序列的不同尺度上取舍， 即是上述所说先筛选合适的KV再算注意力

-   压缩注意力：把控全局信息
    
-   选择注意力：把控局部信息
    
-   滑窗注意力：把控关联紧密信息

![](https://pic4.zhimg.com/v2-3afc35658c74b2b4ed92df8368a8b5cf_1440w.jpg)

脱离原论文公式，在`NSA`的框架图里我们用最简单的数值来描述，计算注意力前我们有$\textbf{q}_{:t}$ 和 $\textbf{k}_{:t},\textbf{v}_{:t}$

1.  词元序列 $t$ 为32，即左上顶部蓝色方格长条每一格代表第$i$ 时刻的$\textbf{k}_{i},\textbf{v}_{i}$
    
2.  将$\textbf{k}_{:32},\textbf{v}_{:32}$ 按照长度为$8$ 划分为4块 $\textbf{k}^{(1)}_{1:8},\textbf{v}^{(1)}_{1:8}$ ,$\textbf{k}^{(2)}_{8:16},\textbf{v}^{(2)}_{8:16}$....
    
3.  **压缩**： 将每个KV块压成一个向量即 $\textbf{k}^{(1)}_{1:8}\rightarrow \tilde{\textbf{k}}^{\text{(1)cmp}}\in \mathcal{R}^{d \times c}$ , $\text{v}^{(1)}$ 同理， 压缩的目的是把一段长度为8的序列压成$c$个序列, 并且有关系$c<8$，为了便于理解我们将$c=1$ 既是1个token代表了一序列。至于怎么压缩我们后面探讨。 这里的压缩注意力是$\textbf{q}_{32}$ 和4个段落的 $\tilde{\textbf{k}}^{\text{(b)cmp}},b={1,2,3,4}$ 进行计算得到4个注意力分数，所以这里做将v进行计算后，就得到压缩的注意力输出了，代表了全局信息$o^{(\text{cmp})}_{32}\in \mathcal{R}^{d \times 1}$
    
4.  **选择**：在压缩时得到的段落注意力分数，选择top-2, 即第2和第4的绿色块。，那么我们就找到对应的段落块当成局部信息$\textbf{k}^{(2)}_{8:16},\textbf{v}^{(2)}_{8:16}$ ,$\textbf{k}^{(4)}_{24:32},\textbf{v}^{(4)}_{24:32}$， 获取到KV后我们也可以计算选择注意力，最终为$o^{(\text{sle})}_{32}\in \mathcal{R}^{d \times 1}$
    
5.  **滑窗**： 在原$\textbf{k}_{:32},\textbf{v}_{:32}$ 序列里取就近的$8$ 个键和值$\textbf{k}_{24:32},\textbf{v}_{24:32}$， 同样可以得到滑窗注意力为$o^{(\text{win})}_{32}\in \mathcal{R}^{d \times 1}$
    
6.  **门控**： 汇聚三种注意力$g^{(cmp)}_{32}o^{(\text{cmp})}_{32}+g^{(sle)}o^{(\text{sle})}_{32}+g^{(win)}o^{(\text{win})}_{32}$, 其中门控$g$ 为输入$x_{32}$ 进行线性层变换并加入sigmoid激活得到。门控是可学习的。
    

至此我们再进一步分析，原来我们有$t=32$个上下文KV，在压缩/选择/滑窗里我们分别有$N_t=4+8*2+8=28$ 个上下文KV，实现了注意力的减少。

当我们的上下文为64k时, 如果我们取128个全局压缩KV，8个512选择块KV和就近窗口4096个KV, 那么我们得到了压缩倍数7.88：

$65536/(128+8\times 512+4096)=65536/8320 \approx 7.88 \\$

那么我们接下来进一步根据原论文描述来实现三种注意力。

### 3.3 压缩注意力实现

压缩注意力的本质是将一段序列的KV压成一个KV，这样就能代表片段的全局信息。

$\tilde{K}^\text{cmp}_t = f_K^\text{cmp}(\mathbf{k}_{:t}) = \left\{\phi(\mathbf{k}_{i d+1: i d+l})\middle| 1\leq i\leq\left\lfloor\frac{t-l}{d}\right\rfloor\right\} \\$

简要描述符号压缩算子$\phi(\cdot):\mathbb{R}^{l\times d}\rightarrow\mathbb{R}^{1\times d}$, 具体的实现可以是一个MLP层和可以学习的位置编码(内部)，MLP参数与数据块长度有关。（文末Appendix.C讨论压缩建模）

> and $\phi$ is a learnable MLP with intra-block position encoding to map keys in a block to a single compressed key.

另外关于KV序列片段的符号：

1.  $t$ : 序列总长度
2.  $l$: 块长度
3.  $d$: 为stride， 目的是使得片段有重叠。当$d=l$ 时无重叠。`NSA`论文的这个符号稍微有些绕。

![](https://pic4.zhimg.com/v2-feed224e169d3d45742385ac7f49b259_1440w.jpg)

我们编程测试当$d \neq l$ , 我们可以跳过这个case

```python
d = 4
max_idx = round(( t - l ) / d)
print(max_idx) # 6
print(torch.arange(max_idx) * d + 1) # tensor([ 1,  5,  9, 13, 17, 21])
print(torch.arange(max_idx) * d + l) # tensor([ 8, 12, 16, 20, 24, 28])
```

另外$d = l$, 并且不减去 $l$ 那么可见这是符合原图的KV 块划分

```python
d = l
max_idx = round(( t ) / d)
print(max_idx) # 4
print(torch.arange(max_idx) * d + 1) # tensor([ 1,  9, 17, 25])
print(torch.arange(max_idx) * d + l) # tensor([ 8, 16, 24, 32])
```

至此我们可以编程实现压缩注意力，给定输入$X \in \mathbb{R}^{\text{bs} \times t \times \text{dim} }$

```python
X = torch.randn(batch_size, t, dim)

Wq = torch.randn(dim, dim)
Wk = torch.randn(dim, dim)
Wv = torch.randn(dim, dim)

Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

提取压缩KV

```python
W_K_cmp = torch.randn(l, 1) #MLP: W2[1,4l]@(W1[4l, l]@X[l, d])
W_V_cmp = torch.randn(l, 1)
W_pe = torch.randn(l, dim)

K_cmp = []
V_cmp = []
for i in range(max_idx):
    cur_K = K[:, i * d + 0: i * d + l , :] + W_pe.unsqueeze(0)
    cur_V = V[:, i * d + 0: i * d + l , :] + W_pe.unsqueeze(0)
    cur_K = cur_K.transpose(1, 2) @ W_K_cmp 
    cur_V = cur_V.transpose(1, 2) @ W_V_cmp
    K_cmp.append(cur_K)
    V_cmp.append(cur_V)

K_cmp = torch.cat(K_cmp, dim = 2).transpose(1,2)
V_cmp = torch.cat(V_cmp, dim = 2).transpose(1,2)
print(K_cmp.shape) # torch.Size([1, 4, 16]) # 长度为32->4
print(V_cmp.shape) # torch.Size([1, 4, 16]) # 长度为32->4
```

再实现多头压缩注意力, 特别要注意：Compression Attention每个头注意到不同的片段。

![](https://pic1.zhimg.com/v2-62f9ffbba3c23d7582b44a1da80890dc_1440w.jpg)

```python
# 多头压缩注意力
Q_mha = Q.view(1, t, heads, head_dim).transpose(1,2)
K_cmp_mha = K_cmp.view(1, block_nums, heads, head_dim).transpose(1,2)
V_cmp_mha = V_cmp.view(1, block_nums, heads, head_dim).transpose(1,2)
score_cmp = Q_mha @ K_cmp_mha.transpose(2,3) # bs, head, q_len, k_cmp_len
print(score_cmp.shape) # torch.Size([1, 4, 32, 4])

p_cmp = F.softmax(score_cmp, dim = -1) # torch.Size([1, 4, 32, 4)
o_cmp = p_cmp @ V_cmp_mha
print(o_cmp.shape) # torch.Size([1, 4, 32, 4]) 

o_cmp = o_cmp.transpose(2, 1).reshape(batch_size, t, dim)
print(o_cmp.shape) # torch.Size([1, 32, 16])
```

在上述代码中重点的是`p_cmp` 维度信息为：\[批次大小,头数 , q序列长度 , 压缩KV序列长度\]

### 3.4 选择注意力实现

在压缩注意力时，我们得到了$q$ 在不同头上，会注意到不同的片段，那么究竟哪个片段我们要再精细的提取特征？我们在压缩注意力阶段得到了，**各个头**(原论文并没有写出来）各个片段的注意力分数

$\mathbf{p}_t^\text{cmp} = \operatorname{Softmax}\left(\mathbf{q}_t^T \tilde{K}_t^\text{cmp}\right), \\$

以下的公式其实是按照带stride的版本写的，看起来很复杂（在文末Appendix. A讨论）

$\mathbf{p}_t^\text{slc}[j] = \sum_{m=0}^{\frac{l'}{d}-1}\sum_{n=0}^{\frac{l}{d} -1} \mathbf{p}_t^\text{cmp}\left[\frac{l'}{d}j+m +n \right], \\$

> 根据原论文描述可以得到非常简单的选择策略。
> 
> When compression blocks and selection blocks share the same blocking scheme, i.e., $l'=l=d$,
> 
> **we can directly obtain the selection block importance scores** $\mathbf{p}_t^\text{slc}$ by $\mathbf{p}_t^\text{slc} = \mathbf{p}_t^\text{cmp}$ straightforwardly.

代入$l'=l=d$ 那么对于该公式，得到下式，直接与压缩注意力切分策略对应，实现就非常简单

$\mathbf{p}_t^\text{slc}[j] = \sum_{m=0}^{\frac{l'}{d}-1}\sum_{n=0}^{\frac{l}{d} -1} \mathbf{p}_t^\text{cmp}\left[\frac{l'}{d}j+m +n \right] = \mathbf{p}_t^\text{cmp}[j] \\$

此时我们可以把多头的压缩注意力分数进行聚合。这就上面提到：压缩注意力不同头，注意到不同的片段

${\mathbf{p}_t^{\text{slc}}}' = \sum_{h=1}^{H} \mathbf{p}_{t}^{\text{slc}, (h)}, \\$

![](https://picx.zhimg.com/v2-1621155aa83d79ed2d77c9e9222a3d6f_1440w.jpg)

编码实现为:

```python
p_slc = p_cmp.sum(dim = 1) # 在head维度上进行合并
print(p_cmp.shape) # torch.Size([1, 4, 32, 4])
print(p_slc.shape) # torch.Size([1, 32, 4])
```

接下来进行选择, 对于不同的 $q$ 有不同的注意片段维度，以下可以选取top-k出来

```python
select_top_k = 2
_, idx = torch.topk(p_slc, dim = 2, k = select_top_k)
print(idx[0,0,:]) # [3,0] 即 q0注意到第3片段和第0片段
idx.shape # [1, 32, 2] : batch_size, q_len, top_k
```

那么我们提取 选择到的片段对应的 KV

```python
idx_slc_start = idx * d
idx_slc_end = idx * d + l
K_slc = torch.randn(batch_size, t, d * select_top_k, dim)
V_slc = torch.randn(batch_size, t, d * select_top_k, dim)
for i in range(batch_size):
    for j in range(t):
        for k in range(select_top_k):
            K_slc[i, j, k * d : k * d + l, :] = K[i, idx_slc_start[i, j, k ] :  idx_slc_end[i, j, k ] , :]
            V_slc[i, j, k * d : k * d + l, :] = V[i, idx_slc_start[i, j, k ] :  idx_slc_end[i, j, k ] , :]
print(K_slc.shape) # bs, seq_len, select_kv, dim, 1,32,16,16, 不同t时刻选到不同的select_kv
print(V_slc.shape) # bs, seq_len, select_kv, dim  1,32,16,16, 不同t时刻选到不同的select_kv
```

上述我们只要选择到KV就可以计算多头注意力了。（在Appendix.B讨论MLA/GQA正交情况）

这里的特征维度为16, 根据MQA和GQA的处理技巧，我们可以共享头，以此减少inference阶段的KV Cache，我们在“内核优化”章节里会描述这种技巧可以减少访存，图示为：

![](https://pic2.zhimg.com/v2-474b8d2b70ce706d0760767246184379_1440w.jpg)

1.  将dim划分成4头4维度
2.  在head维度进行聚合，我们可以写出代码为

```python
# shared head KV
# IN GQA Group: [1-head KV & N-head Q] ----repeat kv-head---> [N-head KV & N-head Q]

V_slc_mha = V_slc.view(batch_size, t, select_top_k * d, heads, head_dim).transpose(2,3)
V_slc = V_slc_mha.sum(dim = 2, keepdim = True)
print(V_slc.shape) # bs, seq_len, head, select_seq_len, head_dim

K_slc_mha = K_slc.view(batch_size, t, select_top_k * d, heads, head_dim).transpose(2,3)
K_slc = K_slc_mha.sum(dim = 2, keepdim = True)
print(V_slc.shape) # bs, seq_len, head, select_seq_len, head_dim
```

最后我们可以计算选择注意力：

注意每个t时刻的单个q要和多个kv计算注意力。

```python
o_slc = torch.zeros(batch_size, t, dim)
for j in range(t):
    Q_mha[:, :, j, :].unsqueeze(dim = 2)
    K_slc_j = K_slc[:, j, :, :, :].repeat(1, heads, 1, 1)
    V_slc_j = V_slc[:, j, :, :, :].repeat(1, heads, 1, 1)
    
    attn_score_j = Q_slc_j @ K_slc_j.transpose(2,3)
    p_slc_j = F.softmax(attn_score_j, dim = -1) 
    # print(p_slc.shape)

    o_slc_j = p_slc_j @ V_slc_j # bs, seq, dim   
    # print(o_slc_j.shape)

    o_slc_j = o_slc_j.transpose(1,2).view(batch_size, 1, dim)
    o_slc[:, j, :] = o_slc_j
print(o_slc.shape)
```

⚠️选择注意力还保留一个细节，即：

1.  注意力形式是GQA
2.  对于一个q，有多头，那么以组的形式，不同的组选到的KV是不一样的。保证了有更多的KV信息来源。
3.  而组内是单头KV，组内共享。保证计算group attention时低SRAM访存的效果。

> where denotes the indexing operator for accessing vector element. For models employing GQA or MQA where key-value caches are shared across query heads, consistent block selection across these heads has to be ensured to minimize KV cache loading during decoding. **The shared importance scores across heads in a group are formally defined as。**

### 3.5 窗口注意力实现

窗口注意力是捕捉与当前q最近的kv片段，这里做了假设，即越相近的KV就越重要，这里补全选择注意力上的“随机性”

这里的实现其实也非常简单，就是提取片段KV

$\tilde{K}_t^\text{win}=\mathbf{k}_{t-w:t}, \tilde{V}_t^\text{win}=\mathbf{v}_{t-w:t} \\$

代码实现为

```python
# built sliding window attention
def get_window_mask(seq_len, window):
    mask = torch.ones(seq_len, seq_len)
    mask = torch.tril(mask)
    win_mask = torch.ones(seq_len - window, seq_len - window)
    win_mask = 1.0 - torch.tril(win_mask)
    mask[window:, :seq_len - window] = win_mask
    return mask
print(get_window_mask(7, 3)) # test
window_mask = get_window_mask(t, 8)
```

检验mask矩阵, 符合预期。

```python
tensor([[1., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0., 0.],
        [0., 1., 1., 1., 0., 0., 0.],
        [0., 0., 1., 1., 1., 0., 0.],
        [0., 0., 0., 1., 1., 1., 0.],
        [0., 0., 0., 0., 1., 1., 1.]])
```

我们就不再赘述多头注意力，快速实现一个单头的版本。

```python
# simplify multihead attention
S = Q @ K.transpose(1,2) / math.sqrt(dim)
S = F.softmax(S, dim = -1)
S = S * window_mask # sliding window mask
print(S)
o_win = S @ V
print(o_win.shape)
```

上述打印, 符合预期. 这里面由于mask规律，实际上我们可以跳过一些块计算, 减少attention计算量

```python
tensor([[[1.0644e-15, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
          0.0000e+00, 0.0000e+00],
         [1.0635e-09, 5.6887e-15, 0.0000e+00,  ..., 0.0000e+00,
          0.0000e+00, 0.0000e+00],
         [1.0207e-12, 5.2067e-10, 3.8002e-10,  ..., 0.0000e+00,
          0.0000e+00, 0.0000e+00],
         ...,
         [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 4.3458e-08,
          0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 4.9893e-18,
          4.5925e-41, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 5.0657e-10,
          1.3605e-06, 3.3918e-06]]])
torch.Size([1, 32, 16])
```

至此我们手撕完了三种注意力，他们的流程均为：

1.  按照规则提取KV Cache片段。
2.  对特定KV片段做特定注意力。
3.  另外特定的片段长度一定原小于长文本长度，比如我们选512维片段长度，而LongContext是64000， 这悬殊的差距，将极大较少矩阵乘，我们再后需讨论如何从内核来优化 稀疏注意力，毕竟不能直接用标准矩阵乘

### 3.6 注意力聚合

在上述三个注意力计算中，我们都得到了同样维度`[1, 32, 16]` 的注意力输出，

这里的门控实现

> $g_t^c\in [0, 1]$ is the gate score for corresponding strategy $c$, derived from input features via an **MLP** and **sigmoid activation.**

```python
W_gated = torch.randn(dim, 3) # mlp, dim->3: cmp, slc, win
gate = X @ W_gated
gate = F.sigmoid(gate) # sigmoid activation
print(gate.shape) # 1, 32, 3 , bs, q_len, gated
```

![](https://pic2.zhimg.com/v2-195dbfbd17a5d3493e2885e9ab43bfe9_1440w.jpg)

那么我们计算完整的`NSA`, $\mathcal{C}$ 代表三种注意力方法

$\mathbf{o}^*_t = \sum_{c \in \mathcal{C}} g_t^c \cdot \text{Attn}(\mathbf{q}_t, \tilde{K}_t^c, \tilde{V}_t^c) \\$

```python
o_list = [o_cmp, o_slc, o_win]
o_star = torch.zeros(batch_size, t, dim)
for i in range(3):
    o_star += gate[:, :, i].unsqueeze(2) * o_list[i]
print(o_star.shape)
```

另外我们知道NSA就是再提取不同片段的KV，可以算提出的KV总量与原KV长度，就能知道减少了多少的计算量

$N_t = \sum_{c \in \mathcal{C}}\text{size}[\tilde{K}^c_t]. \\$

如果原KV长度为$t$ , 那么有${N_t}{} \ll t$.

## 4\. 内核优化

DeepSeek团队重新用triton编写了`NSA`的注意力算子，本质上与Flash-Attention2思路没有大区别，简要描述

> To address this, our key optimization lies in a different query grouping strategy: for each position on the query sequence, we load all query heads within a GQA group (they share the same sparse KV blocks) into SRAM.

**注意力实现**：整体来看的话，NSA实质的Attention计算是GQA，我们以一组注意力来说，涉及到多头Q和单头KV：

-   常规：我们需要将单头KV，复制成与Q头数相对应的KV，送到SRAM计算注意力
-   NSA：单头KV送到SRAM，这里不需要复制多头。

**内核实现**：

1.  Grid Loop：先加载单个多头$Q$ ， 逐个元素来算NSA，注意看维度有 $d_k\times h$ , h为头数量
2.  Inner Loop: 单头K，载入维度为 $d_k\times B_k$ ， $B_k$ 为一个片段，这里并没有头维度，所以从始至终在一个Group里，在HBM/SRAM都是单头KV而存在的。对于一个GQA里的一个Group来说，需要将单头KV复制成多头KV，这里内核优化的关键在于，通过单头按照某种共享内存策略，让多头(multi-grid) 都能访问到一个share的KV，这样就不用复制KV成多头了。

![](https://pic2.zhimg.com/v2-3668a65a434e75595bff042404d21d09_1440w.jpg)

GQA组内单头KV, 减少HBM/SRAM存储和访存时间，SRAM里KV是share memory，不用特地复制成多头

以上的一个绿色块代表了一个q和一段kv计算。这里我们注意到，当$t$ 增大时，由于KV块恒为3块，那么越长NSA的加速越明显。

Selection Attention才是真正体现了Sparse的精髓。下图的注意力计算发生在Selection Attention中，以GQA/MQA视角来看, 一个组内只需要单头KV； 从SRAM视角来看：单头KV可以减少HBM与SRAM的访存量。

![](https://pic4.zhimg.com/v2-9730a3217e974886c6e15c0c665f3777_1440w.jpg)

## 5\. NSA分析

### 5.1 稀疏化注意力机制分析

在上面一通操作下，实现了原生的注意力计算，我们再重新思考注意力的稀疏性：

1.  标准的attention(MHA) 有一定的稀疏性，稀疏性的学习是自动的，但是MHA并没有对极小的注意力分数进行过滤
2.  `NSA` 其实可以看成是分层的注意力学习机制，我们以选择注意力来说：压缩注意力是外层稀疏性建模，选择注意力是内层注意力建模。这里的稀疏性是发生在外层，好处在于只计算几个极小量的$\textbf{(K,V)}^{(\text{cmp})}$ 的注意力就能筛选出局部的精细KV出来。

另外我们要分析稀疏化化的影响，稀疏化如果作用在文本序列上来说是信息是割裂的，这个影响在`NSA`的机制是否严重，我们考虑单层和多层注意力的`NSA` QKV信息量。

1.  单层：非选择到的KV块被丢弃
2.  多层: 下例所示有4个数据块，第一层`NSA`选取到1,3片段，第二层`NSA`选取到2,3 片段， 第三层`NSA`选取到1,4片段，那么可以了解到，经过多层处理这里的数据切割的影响是会被减轻的。

![](https://pic3.zhimg.com/v2-59b1d452b8b683ffba51ccfa4c909f1a_1440w.jpg)

另外我们在”为什么没有Q-Cache“上讨论过，当前的q信息会流转到下一层的KV上，那么就意味着选择哪一个数据块，实际上都有完整的历史的前驱Q信息。即在第一层attention以后，所有流向的特征都已经经过序列建模的。

### 5.2 NSA Inference analysis

**在`NSA`在inference过程中，在计算`NSA`之前，KV-Cache存储实际上是不会减少的。**

另外我们再分析三种注意力是否能内部加入mini-kv-cache

在prefill阶段：

-   压缩注意力：做标准的注意力forward
-   选择注意力：做block-wise的注意力，由于稀疏性原因，可以大幅减轻首token计算时间的原因，
-   窗口注意力：做标准的注意力forward

在decoding阶段

在计算层面，实际上计算量是固定的为单个q 和 固定数量的 KV 进行注意力, 标准的注意力KV是随context累增的。

-   压缩注意力：由于kv是累增的，那么$\textbf{K}^{\text{cmp}}$ 是累增的，可以加入mini-kv-cache
-   选择注意力：复用原KV-Cache
-   窗口注意力：复用原KV-Cache

## 6\. 总结

实验部分里，重点关注精度和速度：

1.  `NSA` 精度优于MHA相当，那么`NSA`大概率是DeepSeek-V4 base的一个核心设计
2.  `NSA` 速度在64k inference相较 Flash Attention 前向加速9倍，反向加速6倍。

实验就不渲染过多，整体方法总结如下：

1.  `NSA`原生稀疏性对于pretrained是必要的，我认为\*\*`NSA`是由inference成本倒推的网络设计\*\*，其关键在于通过减少KV数量(非token数量)来减少注意力的计算复杂度，另外在inference阶段KV-Cache是存储是与原本注意力是一样的，那么这里可以正交MLA，另外随context长度增长，计算量到一个阶段不会累增(MHA计算复杂度随长度平方复杂度递增)
    
2.  标准的Attention本身能学习到序列关系的稀疏性，`NSA`实际上是“人工”来加速“稀疏性”的学习，把`NSA` 看成是**分层**的稀疏性建模，
    
3.  对于稀疏性是否会导致信息缺失，我认为可以在第一层可以做标准的attention建模，这样注意力输出就已经有聚合到每个token序列特征了。后续层层上`NSA`。这样在第二层里哪怕我们选择了片段的KV，那么这个片段在第一层的KV也会有前驱token的信息。

## 7\. 本人手撕NSA代码开源(非官方)

-   纯pytorch实现，无数据依赖，不需要gpu可以运行
-   实现multi-head compress attention
-   实现kv-share-head sparse selection attention
-   实现sliding window attention
-   实现gated native sparse attention

## Reference

GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

Attention Is All You Need

Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

Fast Transformer Decoding: One Write-Head is All You Need

FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

___

## Appendix A： Selction Important Score计算

在3.2里还保留一个复杂的公式，我们试着来编程看看结果

$$$ \mathbf{p}_t^\text{slc}[j] = \sum_{m=0}^{\frac{l'}{d}-1}\sum_{n=0}^{\frac{l}{d} -1} \mathbf{p}_t^\text{cmp}\left[\frac{l'}{d}j+m +n \right], $$$

### A.1: Case 1:

```python
d = 8
t = 512 + d
l_cmp = 16
l_slc = 8 # from paper ： Let l‘ denote the selection block size, or setting l_slc = 32
m_max = l_slc // d
n_max = l_cmp // d
print(m_max) # 1 
print(n_max) # 2

# original is t token, compress -> t_cmp token.
t_cmp = (t - d) // l_cmp
print(t_cmp) # 32

t_cmp = (t - d) // l_cmp

p_cmp = torch.randn(t_cmp)
p_slc = torch.zeros_like(p_cmp)
j_factor = l_slc // d 
```

可以逐`j`计算， `m_max=1` 如果没有外循环的话， 那么 $p_\text{cmp}$ 会取两个片段归并。【扩大了跨片段信息】

-   p\_slc\[j\] = p\_cmp\[j + 0\] + p\_cmp\[j + 1\], n={0, 1}

```python
for j in range(t_cmp):
    for m in range(m_max):
        for n in range(n_max):
            idx = j_factor * j + m + n
            if idx >= t_cmp:
                continue
            else:
                p_slc[j] += p_cmp[idx]

print(p_slc)
```

得到

```python
tensor([-0.2055, -2.1388,  0.6776,  0.4184, -0.0693, -2.1594, -3.6918,  0.6945,
         3.0587,  1.5769,  1.0593, -1.1392, -1.3571, -0.7378,  0.4463,  0.8347,
         0.5641,  1.7130,  0.5760, -0.0429,  1.0820,  3.2252,  2.5116, -0.8499,
        -2.0525,  1.0815,  1.2698, -1.0138, -1.2659, -0.2534,  1.1021,  0.4464])
```

-   上述例子：m=1，n=2, 当`j=0`, 手动展开两个连加符号P\_slc\[0\]=P\_cmp\[0\] + P\_cmp\[1\]，跨两个数据片段

![](https://pic2.zhimg.com/v2-a1a5b00a084e1a8a50ac821f63a35c35_1440w.jpg)

假设例子: m=2, n=2, 当j=0, 手动展开两个连加符号:

P\_slc\[0\]=P\_cmp\[0+0\] + P\_cmp\[0+1\] + P\_cmp\[1+0\] + P\_cmp\[1+1\]，跨3个数据片段，其中一个数据片段重叠连次。

### A.2 Case 2

但是当我们设置为

```text
d = 8
t = 512 + d
l_cmp = 16
l_slc = 16
```

那么， 有一半压缩token没有重要性分数。这里行为奇怪。

```python
tensor([ 0.8175,  4.1195, -0.2032, -2.2682, -0.5053, -1.3227, -6.4006, -1.3999,
         0.2567, -3.1612, -2.3654, -2.6925,  0.6644,  1.0851, -6.9734, -2.6441,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000])
```

## Appendix.B GQA和MLA正交情况分析

问题：如何实现低KV Cache，低SRAM访存的注意力计算方案？

我们考虑更复杂的几种情况的块注意力，

1.  Q头与K头数量相等：可以用标准的MHA来实现注意力计算
2.  Q头与K头数量不相等：比如GQA, 我们在一个group里有单头KV和多头Q，虽然同样在算注意力时要满头，但是有两个好处：a. 单头KV减少对SRAM的访存, b. 在inference阶段减少KV Cache， 前者加速LLM所有阶段(training, prefill, decoding)，后者加速decoding阶段。但是这里的GQA的KV Cache相较MLA来说还是高的。
3.  MLA和GQA正交：比较粗暴的是存MLA latent 当成Cache，再up成单头KV(而不是满头KV)，进入到第2个case，这样就实现了低Cache、低SRAM访存的方案。

## Appendix.C 压缩建模

压缩注意力的本质是将一段序列的KV压成一个KV，这样就能用一个KV代表片段的全局信息。那么究竟怎么压缩更合适？

在论文里定义了压缩算子 $\phi(\cdot)$ 我们来讨论可行性的压缩

1.  非Learning方式：通过mean\_pooling方式来进行压缩
2.  Learning方式：已知K序列 $\textbf{k}\in \mathbb{R}^{l\times \text{dim}}$ 的块大小 $l$ 和维度 $\text{dim}$ , 给定参数矩阵 $W\in \mathbb{R}^{1 \times l}$ , 通过投影变换， $W \textbf{k}=\tilde{\textbf{k}} \in \mathbb{R}^{1 \times d}$ , 可以通过可学习的参数来组合序列，自由度更高.

___

**_《手撕RLHF》_** 解析如何系统的来做LLM对齐工程

[小冬瓜AIGC：【X-R1】 3B中文推理开源, 支持LoRA训练](https://zhuanlan.zhihu.com/p/24292760126)

[小冬瓜AIGC：【手撕LLM-GRPO】你只管给Reward, 剩下的交给RL（附代码）](https://zhuanlan.zhihu.com/p/20812786520)

[小冬瓜AIGC：再深挖DeepSeek-R1: Reward is Enough](https://zhuanlan.zhihu.com/p/20053834500)

[小冬瓜AIGC：【解读】DeepSeek-R1：RL前真的不需要SFT了吗???](https://zhuanlan.zhihu.com/p/19623772462)

[小冬瓜AIGC：【OpenAI o3安全对齐方案】坏消息：RLHF里的HF无了!!](https://zhuanlan.zhihu.com/p/14792481053)

[小冬瓜AIGC：【o1推理】Scaling LLM Test-Time：谁说类o1推理一定要用RL?!](https://zhuanlan.zhihu.com/p/877197813)

[小冬瓜AIGC：为什么DPO里Chosen和Rejected概率会同时下降???](https://zhuanlan.zhihu.com/p/6327313416)

[小冬瓜AIGC：【手撕RLHF-DPO】step-by-step公式推导及实验分析](https://zhuanlan.zhihu.com/p/692991235)

[小冬瓜AIGC：【手撕RLHF-Aligner】7B模型外挂，暴涨GPT4安全性26.9%](https://zhuanlan.zhihu.com/p/682627363)

[小冬瓜AIGC：【手撕RLHF\_Weak-to-Strong】OpenAI超级对齐新思路（含代码解析）](https://zhuanlan.zhihu.com/p/674714374)

[小冬瓜AIGC：【手撕RLHF-Safe RLHF】带着脚镣跳舞的PPO](https://zhuanlan.zhihu.com/p/670288679)

[小冬瓜AIGC：【手撕RLHF-Rejection Sampling】如何优雅的从SFT过渡到PPO](https://zhuanlan.zhihu.com/p/669397860)

[小冬瓜AIGC：【手撕RLHF-LLaMA2】 Reward Model PyTorch实现](https://zhuanlan.zhihu.com/p/679012951)

**_《手撕LLM》_**系列文章+原创课程：LLM原理涵盖Pretrained/PEFT/RLHF/高性能计算

[小冬瓜AIGC：【手撕LLM\_Nv-Embed】英伟达LLM-as-Embedding, ICLR高分佳作, RAG检索有救了!!!](https://zhuanlan.zhihu.com/p/16854104123)

[小冬瓜AIGC：【手撕LLM-Cut Cross Entropy】ICLR高分：LLM训练交叉熵的Memory-Efficient优化](https://zhuanlan.zhihu.com/p/13548439339)

[小冬瓜AIGC：【手撕online softmax】Flash Attention前传，一撕一个不吱声](https://zhuanlan.zhihu.com/p/5078640012)

[小冬瓜AIGC：【手撕LLM-FlashAttention2】只因For循环优化的太美](https://zhuanlan.zhihu.com/p/670085985)

[小冬瓜AIGC：【手撕LLM-Flash Attention】从softmax说起，保姆级超长文！！](https://zhuanlan.zhihu.com/p/663932651)

[小冬瓜AIGC：【手撕LLM】长文本的Position Encoding的衰减性证明](https://zhuanlan.zhihu.com/p/709234529)

[小冬瓜AIGC：【手撕LLM-NTK RoPE】长文本“高频外推、低频内插“从衰减性视角理解](https://zhuanlan.zhihu.com/p/702964625/edit)

[小冬瓜AIGC：【手撕LLM - Mixtral-8x7B】Pytorch 实现](https://zhuanlan.zhihu.com/p/680361287)

[小冬瓜AIGC：【手撕LLM-Medusa】并行解码范式: 美杜莎驾到, 通通闪开！！](https://zhuanlan.zhihu.com/p/686000524)

[小冬瓜AIGC：【手撕LLM-Speculative Decoding】大模型迈向"并行"解码时代](https://zhuanlan.zhihu.com/p/671432448)

[小冬瓜AIGC: 【手撕LLM-Generation】Top-K+重复性惩罚](https://zhuanlan.zhihu.com/p/667025336)

[小冬瓜AIGC：【手撕LLM-KVCache】显存刺客的前世今生--文末含代码](https://zhuanlan.zhihu.com/p/667763542)

___

> 小冬瓜AIGC | X-R1开源框架 | 现高校LLM对齐研究
> 
> 原创课程帮助学员拿下OpenAI, Meta, 字节SEED等
