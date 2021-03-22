# Transformer-XL论文笔记
解决问题：如何赋予编码器捕获长距离依赖的能力

## Motivation
Transformer编码固定长度的上下文，即将一个长的文本序列截断为几百个字符的固定长度片段(segment)，然后分别编码每个片段[1]，片段之间没有任何的信息交互。比如BERT，序列长度的极限一般在512。动机总结如下：

Transformer无法建模超过固定长度的依赖关系，对长文本编码效果差。
Transformer把要处理的文本分割成等长的片段，通常不考虑句子（语义）边界，导致上下文碎片化(context fragmentation)。通俗来讲，一个完整的句子在分割后，一半在前面的片段，一半在后面的片段。
文章围绕如何建模长距离依赖，提出Transformer-XL【XL是extra long的意思】：

提出片段级递归机制(segment-level recurrence mechanism)，引入一个记忆(memory)模块（类似于cache或cell），循环用来建模片段之间的联系。
使得长距离依赖的建模成为可能；
使得片段之间产生交互，解决上下文碎片化问题。
提出相对位置编码机制(relative position embedding scheme)，代替绝对位置编码。
在memory的循环计算过程中，避免时序混淆【见model部分】，位置编码可重用。
小结一下，片段级递归机制为了解决编码长距离依赖和上下文碎片化，相对位置编码机制为了实现片段级递归机制而提出，解决可能出现的时序混淆问题。

## segment-level recurrence mechanism
为了解决长距离依赖，文章引入一个memory状态。

在训练过程中，每个片段的表示为最后的隐层状态​，​表示片段的序号，​表示片段的长度，​表示隐层维度。

在计算​片段的表示时，用memory缓存​片段​层的隐层状态​，用来更新​，这样就给下一个片段同了上文，长距离依赖也通过memory保存了下来。并且，最大可能的依赖长度线性增长，达到 [公式] 。
![](https://cdn.jsdelivr.net/gh/donttal/figurebed/img/Transformer-XL.png)

## relative position embedding scheme
按照绝对位置向量的算法，两个片段之间同一个词语所有的位置向量是一样的，因此它们的表示向量是一样的，难以区分。
$$\begin{aligned} \mathbf{A}_{i, j}^{\mathrm{rl}} &=\underbrace{\mathbf{E}_{x_{i}}^{\top} \mathbf{W}_{q}^{\top} \mathbf{W}_{k, E} \mathbf{E}_{x_{j}}}_{(a)}+\underbrace{\mathbf{E}_{x_{i}}^{\top} \mathbf{W}_{q}^{\top} \mathbf{W}_{k, R} \mathbf{R}_{i-j}}_{(b)} \\ &+\underbrace{u^{\top} \mathbf{W}_{k, E} \mathbf{E}_{x_{j}}}_{(c)}+\underbrace{v^{\top} \mathbf{W}_{k, R} \mathbf{R}_{i-j}}_{(d)} \end{aligned}$$
1. 引入相对位置编码，用的是Transformer里用的sinusoid encoding matrix，不需要学。
2. u和v是需要学习的参数，这是这部分的关键。在计算self-attention时，由于query所有位置对应的query向量是一样的，因此不管的query位置如何，对不同单词的attention偏差应保持相同。
3. $\mathbf{W}_{k, E}$,$\mathbf{W}_{k, R}$也是需要学习的参数，分别产生基于内容的key向量和基于位置的key向量。

## faster evaluation
在评估时， Transformer-XL比Vanilla Transformer具有更长的有效上下文，并且Transformer-XL能够在不需要重新计算的情况下处理新段中的所有元素，显著提高了速度。
特别是对于较长的上下文。例如，对于 800 个字符的上下文长度，Transformer-XL 比Vanilla Transformer 快 363 倍；而对于 3800 字符的上下文，Transformer-XL 快了 1874 倍。

## Reference
[1]. Zihang Dai, Zhilin Yang, Yiming Yang, William W Cohen, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. [Transformer-xl: Attentive language models beyond a ﬁxed-length context.](https://arxiv.org/abs/1901.02860.pdf) arXiv preprint arXiv:1901.02860, 2019.

[2]. 机器之心报道. https://zhuanlan.zhihu.com/p/56027916.

[3]. 官方代码. https://github.com/kimiyoung/transformer-xl.

[4]. Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov and Quoc V. Le. [XLNet: Generalized Autoregressive Pretraining for Language Understanding.](https://arxiv.org/abs/1906.08237.pdf) arXiv preprint arXiv:1906.08237, 2019.
