# BERT+知识图谱：北大-腾讯联合推出知识赋能的K-BERT模型
[参考网址](https://mp.weixin.qq.com/s/9Efq9qMKJBz0DQH3vveqdA)
当阅读特定领域文本时，普通人只能根据其上下文理解单词，而专家则可以利用相关领域知识进行推断。目前公开的 BERT、GPT、XLNet 等预训练模型均是在开放领域语料预训练得到的，其就像一个普通人，虽然能够读懂通用文本，但是对于专业领域文本时却缺乏一定的背景知识。

解决这一问题的一个方法是使用专业语料预训练模型，但是预训练的过程是十分耗时和耗计算资源的，普通研究者通常难以实现。例如，如果我们希望模型获得“扑热息痛可以治疗感冒”的知识，则在训练语料库中需要大量同时出现“扑热息痛”和“感冒”的句子。不仅如此，通过领域语料预训练的方式引入专家知识，其可解释性和可控性较差。


除了以上策略，我们还能做些什么来使模型成为领域专家？知识图谱（Knowledge Graph，KG）是一个很好的解决方案。

随着知识细化为结构化形式，许多领域的 KG 都被构建起来，例如，医学领域的 SNOMED-CT，中国概念的 HowNet。如果 KG 可以集成到预训练语言模型中，它将为模型配备领域知识，从而提高模型在特定领域任务上的性能，同时降低大规模的预训练成本。此外，知识图谱具有很高的可解释性，因为可以手动编辑注入的知识。


目前，将知识图谱与语言模型结合的研究有哪些呢？最具代表性的就是清华的 ERNIE，其使用一个独立的 TransE 算法获得实体向量，然后再将实体向量嵌入到 BERT 中。清华 ERNIE 的工作很有借鉴意义，但是仍然存在一些可改进的地方，例如：

**1. 知识图谱中的关系信息没有被用到；**

**2. 实体向量和词向量是使用不同的方法得到的，可能存在空间的不一致；**

**3. 对于实体数量巨大的大规模知识图谱，实体向量表将占据很大的内存。**

另外，将过多的知识引入到语言表示模型中，可能会改变原来句子的含义，本文称为知识噪声问题。为了解决以上问题，本文的研究人员尝试不区分实体向量和词向量，使用统一的向量空间将知识注入语言表示模型中。

## 方法
基于以上想法，研究人员对 Google BERT 进行了一定的改进，提出了 K-BERT 模型。下面介绍 K-BERT 的具体思想，图 1 是 K-BERT 的总体架构图。
![](https://cdn.jsdelivr.net/gh/donttal/figurebed/img/KBERT.png)

当一个句子“Tim Cook is currently visiting Beijing now”输入时，首先会经过一个知识层（Knowledge Layer），知识层将知识图谱中关联到的三元组信息（Apple-CEO-Tim Cook、Beijing-capital-China 等）注入到句子中，形成一个富有背景知识的句子树（Sentence tree）。

可以看出，通过知识层，一个句子序列被转换成了一个树结构或图结构，其中包含了句子中原本没有的背景知识，即我们知道“苹果的 CEO 现在在中国”。

得到了句子树以后，问题出现了。传统的 BERT 类模型，只能处理序列结构的句子输入，而图结构的句子树是无法直接输入到 BERT 模型中的。如果强行把句子树平铺成序列输入模型，必然造成结构信息的丢失。在这里，K-BERT 中提出了一个很巧妙的解决办法，那就是软位置（Soft-position）和可见矩阵（Visible Matrix）。下面我们详细看看具体的实现方法。

众所周知，在 BERT 中将句子序列输入到模型之前，会给句子序列中的每个 token 加上一个位置编码，即 token 在句子中的位次，例如“Tim(0) Cook(1) is(2) currently(3) visiting(4) Beijing(5) now(6)”。如果没有位置编码，那 BERT 模型是没有顺序信息的，相当于一个词袋模型。

在 K-BERT 中，首先会将句子树平铺，例如图 2 中的句子树平铺以后是“[CLS] Tim Cook CEO Apple is currently visiting Beijing capital China is_a City now”。

![](https://cdn.jsdelivr.net/gh/donttal/figurebed/img/位置编码.png)

显然，平铺以后的句子是杂乱不易读的，K-BERT 通过软位置编码恢复句子树的顺序信息，即“[CLS](0) Tim(1) Cook(2) CEO(3) Apple(4) is(3) visiting(4) Beijing(5) capital(6) China(7) is_a(6) City(7) now(6)”,可以看到“CEO(3)”和“is(3)”的位置编码都 3，因为它们都是跟在“Cook(2)”之后。

只用软位置还是不够的，因为会让模型误认为 Apple (4) 是跟在 is (3) 之后，这是错误的。K-BERT 中最大的亮点在于 Mask-Transformer，其中使用了可见矩阵（Visible matrix）将图或树结构中的结构信息引入到模型中。

回顾一下 BERT 中 Self-attention，一个词的词嵌入是来源于其上下文。Mask-Transformer 核心思想就是让一个词的词嵌入只来源于其同一个枝干的上下文，而不同枝干的词之间相互不影响。这就是通过可见矩阵来实现的，图 2 中的句子树对应的可见矩阵如图 3 所示，其中一共有 13 个 token，所以是一个 13*13 的矩阵，红色表示对应位置的两个 token 相互可见，白色表示相互不可见。

![](https://cdn.jsdelivr.net/gh/donttal/figurebed/img/可见矩阵.png)

有了可见矩阵以后，可见矩阵该如何使用呢？其实很简单，就是 Mask-Transformer。对于一个可见矩阵 M，相互可见的红色点取值为 0，相互不可见的白色取值为负无穷，然后把 M 加到计算 self-attention 的 softmax 函数里就好，即如下公式。

![](https://cdn.jsdelivr.net/gh/donttal/figurebed/img/位置编码公式.png)

以上公式只是对 BERT 里的 self-attention 做简单的修改，多加了一个 M，其余并无差别。如果两个字之间相互不可见，它们之间的影响系数 S[i,j] 就会是 0，也就使这两个词的隐藏状态 h 之间没有任何影响。这样，就把句子树中的结构信息输入给 BERT 了。

![](https://cdn.jsdelivr.net/gh/donttal/figurebed/img/知识噪声.png)

总结一下，Mask-Transformer 接收句子树作为输入的过程如图 5。

![](https://cdn.jsdelivr.net/gh/donttal/figurebed/img/句子树.png)

其实就是对应了原论文中的结构图，如图 6，对于一个句子树，分别使用 Token 序列保存内容，用可见矩阵保存结构信息

![](https://cdn.jsdelivr.net/gh/donttal/figurebed/img/处理过程.png)

从图 6 中可以看出，除了软位置和可见矩阵，其余结构均与 Google BERT 保持一致，这就给 K-BERT 带来了一个很好的特性——兼容 BERT 类的模型参数。K-BERT 可以直接加载 Google BERT、Baidu ERNIE、Facebook RoBERTa 等市面上公开的已预训练好的 BERT 类模型，无需自行再次预训练，给使用者节约了很大一笔计算资源。

## 实验结果
下面我们来看看 K-BERT 的实验效果。首先，本文采用了三个知识图谱，分别是 CN-DBpedia、知网（HowNet）和自建的医学知识图（MedicalKG）。用于测评的任务分为两类，分别是开放领域任务和专业领域任务。开放领域任务一共有 8 个，分别是 Book review、Chnsenticorp、Shopping、Weibo、XNLI、LCQMC、NLPCC-DBQA、MSRA-NER，实验结果如下表。

![](https://cdn.jsdelivr.net/gh/donttal/figurebed/img/结果表.png)

可以看出，K-BERT 相比于 Google BERT，在开放领域的任务上有一点微小的提升，但是提升不是很明显。可能的原因在于开放领域的任务并不需要背景知识。

为了测试在需要“背景知识”的任务上的效果，研究者使用了四个特定领域的任务，分别是金融问答、法律问答、金融实体识别和医学实体识别。实验效果见下图。

可以看出，在特定领域任务上的表现还是不错的，这些特定领域任务对背景知识的要求较高。总体而言，知识图谱适合用于提升需要背景知识的任务，而对于不需要背景知识的开放领域任务往往效果不是很显著。

![](https://cdn.jsdelivr.net/gh/donttal/figurebed/img/结果.png)

目前，本工作已被 AAAI-2020 收录。研究者还指出，目前 K-BERT 还存在很多问题需要被解决，例如：当知识图谱质量过差时如何提升模型的鲁棒性；在实体关联时如何剔除因一词多义造成的错误关联。研究者希望将结构化的知识图谱引入到 NLP 社区中，目前还需要做很多努力。K-BERT 还不够完善，将来还会不断更新，欢迎大家关注

## 后记

K-BERT的代码已开源，论文原文和项目地址如下： 

**论文地址：https://arxiv.org/abs/1909.07606v1 **
**项目地址：https://github.com/autoliuweijie/K-BERT **

如果你对自然语言处理、知识图谱感兴趣，希望从事这方面的研究，欢迎与我们联系。 

联系邮箱：rickzhou@tencent.com
联系邮箱：nlpzhezhao@tencent.com


## 参考文献

[1] Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2018. BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805. 

[2] Zhang, Z.; Han, X.; Liu, Z.; Jiang, X.; Sun, M.; and Liu, Q. 2019. ERNIE: Enhanced language representation with informative entities. arXiv preprint arXiv:1905.07129. 

[3] Xu, B.; Xu, Y.; Liang, J.; Xie, C.; Liang, B.; Cui, W.; and Xiao, Y. 2017. Cn-dbpedia: A never-ending chinese knowl- edge extraction system. International conference industrial, engineering and other applications applied intelligent sys- tems 428–438. 

[4] Dong, Z.; Dong, Q.; and Hao, C. 2006. Hownet and the computation of meaning.