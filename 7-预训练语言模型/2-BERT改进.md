## BERT系列模型进展介绍
[toc]


#### Q1 生成任务 MASS/UNILM 

首先，对于第一个课题： **如何将 BERT 用于生成任务。** 从技术上来说， Encoder-Decoder 架构应该是首选的框架了， Encoder 输入原句子，Decoder 生成新句子，那么问题在于，Encoder 与 Decoder 如何表示？

对于 Encoder 端来说，我们只需要将 Bert 直接初始化就行；那么对于Decoder 端呢？ 也采用 Bert 初始化吗？ 要知道的是， Decoder 可是用来生成的， 如果你的 embedding 信息是通过 AE 模型训练得到的，那么生成效果估计会诡异的一批。 那么现在的问题就变成了， **如何合理的初始化 Decoder 端的 embedding 信息呢？**

然后，我们再来谈谈第二个课题：**如何设计一个适合于生成任务的语言模型。** 目前从我看到的两篇文章中有两个思路：

- MASS 通过 mask 连续的**一小段**来试图即学习到理解知识，又学习到生成知识， **通过预测一段连续的 tokens 的确有助于提高模型生成方面的能力**，但我个人觉得 mask 一小段信息所提升的生成能力十分有限， 且我认为这会影响到模型理解方面的能力。
- UULM 就厉害了， 它涉及了一组语言模型： **Unidirectional LM， Masked Bidirectional LM， Seq2Seq LM**

**1、MASS(微软)**[[15\]](https://zhuanlan.zhihu.com/p/76912493#ref_15)

![img](https://pic4.zhimg.com/80/v2-8aa755819300e8a3c0d36dd73368c517_hd.jpg)

- - - 统一预训练框架:通过类似的Seq2Seq框架，在预训练阶段统一了BERT和LM模型；
    - Encoder中理解unmasked tokens；Decoder中需要预测连续的[mask]tokens，获取更多的语言信息；Decoder从Encoder中抽取更多信息；
    - 当k=1或者n时，MASS的概率形式分别和BERT中的MLM以及GPT中标准的LM一致（k为mask的连续片段长度））

**2、UNILM (微软)**[[16\]](https://zhuanlan.zhihu.com/p/76912493#ref_16)**：**

![img](https://pic4.zhimg.com/80/v2-200519af01d9eb8f345f3e7d3d059303_hd.jpg)

- - - 统一预训练框架:和直接从mask矩阵的角度统一BERT和LM；
    - 3个Attention Mask矩阵：LM、MLM、Seq2Seq LM；
    - 注意：UNILM中的LM并不是传统的LM模型，仍然是通过引入[MASK]实现的；
    

#### Q2：引入知识 

**1、ERNIE 1.0 (百度)**[[17\]](https://zhuanlan.zhihu.com/p/76912493#ref_17)**：**

- - 在预训练阶段引入知识（实际是预先识别出的实体），引入3种[MASK]策略预测：

  - - Basic-Level Masking： 跟BERT一样，对subword进行mask，无法获取高层次语义；
    - Phrase-Level Masking： mask连续短语；
    - Entity-Level Masking： mask实体；

**2、ERNIE (THU)**[[18\]](https://zhuanlan.zhihu.com/p/76912493#ref_18)**：**

![img](https://pic2.zhimg.com/80/v2-faaa66f6d2c0ce8fa9998f21e97dc691_hd.jpg)

- - 基于BERT预训练原生模型，将文本中的实体**对齐**到外部的知识图谱，并通过知识嵌入得到实体向量作为ERNIE的输入；
  - 由于语言表征的预训练过程和知识表征过程有很大的不同，会产生两个独立的向量空间。为解决上述问题，在有实体输入的位置，将实体向量和文本表示通过**非线性变换进行融合**，以融合词汇、句法和知识信息；
  - 引入改进的预训练目标 **Denoising entity auto-encoder** (DEA)：要求模型能够根据给定的实体序列和文本序列来预测对应的实体；

#### **Q3 多任务学习机制**

**多任务学习**(Multi-task Learning)[[19\]](https://zhuanlan.zhihu.com/p/76912493#ref_19)是指同时学习多个相关任务，让这些任务在学习过程中共享知识，利用多个任务之间的相关性来改进模型在每个任务的性能和泛化能力。多任务学习可以看作是一种归纳迁移学习，即通过利用包含在相关任务中的信息作为归纳偏置(Inductive Bias)来提高泛化能力。多任务学习的训练机制分为同时训练和交替训练。

**1、MTDNN(微软)**[[20\]](https://zhuanlan.zhihu.com/p/76912493#ref_20)**：**在下游任务中引入多任务学习机制

![img](https://pic1.zhimg.com/80/v2-803736918db62a42c1be08d5e7ec10d4_hd.jpg)

**2、ERNIE 2.0 (百度)**[[21\]](https://zhuanlan.zhihu.com/p/76912493#ref_21)**：**

![img](https://pic1.zhimg.com/80/v2-f9f5c4046ef6a1afa829cb3f726d477c_hd.jpg)

- - MTDNN是在下游任务引入多任务机制的，而ERNIE 2.0 是在预训练引入多任务学习（与先验知识库进行交互），使模型能够从不同的任务中学到更多的语言知识。

  - 构建多个层次的任务全面捕捉训练语料中的词法、结构、语义的潜在知识。主要包含3个方面的任务：

  - - 词法层面，word-aware 任务：捕捉词汇层面的信息，如英文大小写预测；
    - 结构层面，structure-aware 任务：捕捉句法层面的信息，如句子顺序问题、句子距离问题；
    - 语义层面，semantic-aware 任务：捕捉语义方面的信息，如语义逻辑关系预测（因果、假设、递进、转折）；

  - 主要的方式是构建**增量学习（**后续可以不断引入更多的任务**）**模型，通过多任务学习**持续更新预训练模型**，这种**连续交替**的学习范式**不会使模型忘记之前学到的语言知识**。

  - - - 将3大类任务的若干个子任务一起用于训练，引入新的任务时会将继续引入之前的任务，防止忘记之前已经学到的知识，具体是一个**逐渐增加任务数量**的过程[[22\]](https://zhuanlan.zhihu.com/p/76912493#ref_22)：
        (task1)->(task1,task2)->(task1,task2,task3)->...->(task1，task2,...,taskN)，

#### Q4 mask策略

原生BERT模型：按照subword维度进行mask，然后进行预测；局部的语言信号，缺乏全局建模的能力。

- BERT WWM(Google)：按照whole word维度进行mask，然后进行预测；

- ERNIE等系列：建模词、短语、实体的完整语义：通过先验知识将知识（短语、词、实体）进行整体mask；引入外部知识，按照entity维度进行mask，然后进行预测；

- SpanBert：不需要按照先验的词/实体/短语等边界信息进行mask，而是采取随机mask：

- - **采用Span Masking**：根据几何分布，随机选择一段空间长度，之后再根据均匀分布随机选择起始位置，最后按照长度mask；通过采样，平均被遮盖长度是3.8 个词的长度；
  - **引入Span Boundary Objective**：新的预训练目标旨在使被mask的Span 边界的词向量能学习到 Span中被mask的部分；新的预训练目标和MLM一起使用；

- 注意：BERT WWM、ERNIE等系列、SpanBERT旨在**隐式地学习**预测词（mask部分本身的强相关性）之间的关系[[23\]](https://zhuanlan.zhihu.com/p/76912493#ref_23)，而在 XLNet 中，是通过 PLM 加上自回归方式来**显式地学习**预测词之间关系；

#### Q5 精细调参

**RoBERTa(FaceBook):**[[24\]](https://zhuanlan.zhihu.com/p/76912493#ref_24)

- - 丢弃NSP，效果更好；
  - 动态改变mask策略，把数据复制10份，然后统一进行随机mask；
  - 对学习率的峰值和warm-up更新步数作出调整；
  - 在更长的序列上训练： 不对序列进行截短，使用全长度序列；


#### Q6 中文领域 BERT-WWM
[BERT-WWM](<https://github.com/ymcui/Chinese-BERT-wwm>)

对于中文领域，分词还是分字一直是一个问题，那么，到底是选分词，还是分字，这一直是一个大问题。 

BERT 无疑选择了分字这条路， ERNIE 通过融入知识，其实带来了部分分词的效果，那么在预训练语言模型中，分词到底有没有用， BERT-WWM 给出了答案。

通过采用 mask 词的方式， 在原有的 BERT-base 模型上接着进行训练， 这其实有种 词 + 字 级别组合的方式， 我在 [深度学习时代，分词真的有必要吗](<https://zhuanlan.zhihu.com/p/66155616>) 中就有提到 字级别 与 词级别之间的差别， 而预训练语言模型能很好的组织二者，的确是件大喜事。

而事实证明， BERT-WWM 在中文任务上的确有着优势所在，具体就不细说了，至少目前来说，我们的中文预训练语言模型有三大选择了： BERT , ERNIE, BERT-WWM。

## XLNet的内核机制探究

在BERT系列模型后，Google发布的XLNet在问答、文本分类、自然语言理解等任务上都大幅超越BERT；XLNet的提出是**对标准语言模型（自回归）的一个复兴**[[25\]](https://zhuanlan.zhihu.com/p/76912493#ref_25)，提出一个框架来连接语言建模方法和预训练方法

**Q16：XLNet**[[26\]](https://zhuanlan.zhihu.com/p/76912493#ref_26)**提出的背景是怎样的？**

- 对于ELMO、GPT等预训练模型都是基于传统的语言模型（自回归语言模型AR），自回归语言模型天然适合处理生成任务，但是无法对双向上下文进行表征，因此人们反而转向自编码思想的研究（如BERT系列模型）；

- 自编码语言模型（AE）虽然可以实现双向上下文进行表征，但是：

- - BERT系列模型引入独立性假设，没有考虑预测[MASK]之间的相关性；
  - MLM预训练目标的设置造成预训练过程和生成过程不一致；
  - 预训练时的[MASK]噪声在finetune阶段不会出现，造成两阶段不匹配问题；

- 有什么办法能构建一个模型使得同时具有AR和AE的优点并且没有它们缺点呢？

### 内核机制分析

![img](https://pic3.zhimg.com/80/v2-41ed57154fef8781103230d29ac4d4ca_hd.jpg)

**1、排列语言模型**（Permutation LM，PLM）：

如果衡量序列中被建模的依赖关系的数量，标准的LM可以达到上界，不像MLM一样，LM不依赖于任何独立假设。借鉴 NADE[[27\]](https://zhuanlan.zhihu.com/p/76912493#ref_27)的思想，XLNet将标准的LM推广到PLM。

- - 为什么PLM可以实现双向上下文的建模？

  - - **PLM的本质就是LM联合概率的多种分解机制的体现**；
    - 将LM的顺序拆解推广到**随机拆解**，但是需要保留每个词的原始位置信息（**PLM只是语言模型建模方式的因式分解/排列，并不是词的位置信息的重新排列！**）
    - 如果遍历 𝑇! 种分解方法，并且模型参数是共享的，**PLM就一定可以学习到各种双向上下文；**换句话说，当我们把所有可能的𝑇! 排列都考虑到的时候，对于预测词的所有上下文就都可以学习到了！
    - 由于遍历 𝑇! 种路径计算量非常大（对于10个词的句子，10!=3628800）。因此实际只能随机的采样𝑇!里的部分排列，并求期望；

![img](https://pic4.zhimg.com/80/v2-c77656a3d92d8cedbf01be32611fb087_hd.jpg)

**2、Two-Stream Self-Attention**

如果采取标准的Transformer来建模PLM，会出现**没有目标(target)位置信息的问题**。问题的关键是模型并不知道要预测的到底是哪个位置的词，从而导致**具有部分排列下的PLM在预测不同目标词时的概率是相同的**。

。

![img](https://pic3.zhimg.com/80/v2-166f69e4355df7741d6e8f25c030df6e_hd.jpg)

- - 怎么解决没有目标(target)位置信息的问题？

  - - 对于没有目标位置信息的问题，XLNet 引入了Two-Stream Self-Attention：

![img](https://pic3.zhimg.com/80/v2-6b794eb9930a1059d518397d96d20d56_hd.jpg)

- - - Query 流就为了预测当前词，只包含位置信息，不包含词的内容信息；
    - Content 流主要为 Query 流提供其它词的内容向量，包含位置信息和内容信息；

## Transformer-XL长文本建模

- BERT(Transformer)的最大输入长度为512，那么怎么对文档级别的文本建模？

- - vanilla model进行Segment，但是会存在上下文碎片化的问题（无法对连续文档的语义信息进行建模），同时推断时需要重复计算，因此推断速度会很慢；

- Transformer-XL改进

- - 引入**recurrence mechanism**(不采用BPTT方式求导)：

  - - 前一个segment计算的representation被修复并缓存，以便在模型处理下一个新的segment时作为扩展上下文resume；
    - 最大可能依赖关系长度增加了N倍，其中N表示网络的深度；
    - 解决了上下文碎片问题，为新段前面的token提供了必要的上下文；
    - 由于不需要重复计算，Transformer-XL在语言建模任务的评估期间比vanilla Transformer快1800+倍；

  - **引入相对位置编码方案：**

  - - 对于每一个segment都应该具有不同的位置编码，因此Transformer-XL采取了相对位置编码；

![img](https://pic3.zhimg.com/80/v2-8b6148634cc8a5656e579dcc9c068dde_hd.jpg)

![img](https://pic1.zhimg.com/80/v2-65c36894a6b3478f13ea548e1352dcb0_hd.jpg)

## 模型压缩

