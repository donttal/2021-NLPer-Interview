# tranformer原理和实现
[TOC]

# Transformer(知乎版本)

![](https://cdn.jsdelivr.net/gh/donttal/figurebed/img/transformer模型结构.jpg)

和经典的 seq2seq 模型一样，Transformer 模型中也采用了 encoer-decoder 架构。上图的左半边用 **NX** 框出来的，就代表一层 encoder，其中论文里面的 encoder 一共有6层这样的结构。上图的右半边用 **NX** 框出来的，则代表一层 decoder，同样也有6层。

定义输入序列首先经过 [word embedding](https://easyai.tech/ai-definition/word-embedding/)，再和 positional encoding 相加后，输入到 encoder 中。

![img](https://pic2.zhimg.com/80/v2-1cfd35f0ff43407e25da3ab25631f82d_hd.jpg)

输出序列经过的处理和输入序列一样，然后输入到 decoder。

最后，decoder 的输出经过一个线性层，再接 Softmax。

## 将张量引入图景

每个单词都被嵌入为512维的向量，我们用这些简单的方框来表示这些向量。

词嵌入过程只发生在最底层的编码器中。所有的编码器都有一个相同的特点，即它们接收一个向量列表，列表中的每个向量大小为512维。在底层（最开始）编码器中它就是词向量，但是在其他编码器中，它就是下一层编码器的输出（也是一个向量列表）。向量列表大小是我们可以设置的超参数——<font color=red>一般是我们训练集中最长句子的长度。</font>

## Encoder

encoder由 6 层相同的层组成，每一层分别由两部分组成：

- 第一部分是 multi-head self-attention
- 第二部分是 position-wise feed-forward network，是一个全连接层

两个部分，都有一个残差连接(residual connection)，然后接着一个 Layer Normalization。

## **Decoder**

解码器的子层也是这样样的。如果我们想象一个2 层编码-解码结构的transformer，它看起来会像下面这张图一样：

![img](https://pic4.zhimg.com/80/v2-12e11c0fea79bc485a6d9f4a2cb12f7f_hd.jpg)

和 encoder 类似，decoder 也是由6个相同的层组成，每一个层包括以下3个部分:

- 第一个部分是 multi-head self-attention mechanism
- 第二部分是 multi-head context-attention mechanism
- 第三部分是一个 position-wise feed-forward network

和 encoder 一样，上面三个部分的每一个部分，都有一个残差连接，后接一个 **Layer Normalization**。

decoder 和 encoder 不同的地方在 multi-head context-attention mechanism

### **解码组件**

既然我们已经谈到了大部分编码器的概念，那么我们基本上也就知道解码器是如何工作的了。但最好还是看看解码器的细节。

编码器通过处理输入序列开启工作。顶端编码器的输出之后会变转化为一个包含向量K（键向量）和V（值向量）的注意力向量集 。这些向量将被每个解码器用于自身的“编码-解码注意力层”，而这些层可以帮助解码器关注输入序列哪些位置合适

在完成编码阶段后，则开始解码阶段。解码阶段的每个步骤都会输出一个输出序列（在这个例子里，是英语翻译的句子）的元素

接下来的步骤重复了这个过程，直到到达一个特殊的终止符号，它表示transformer的解码器已经完成了它的输出。每个步骤的输出在下一个时间步被提供给底端解码器，并且就像编码器之前做的那样，这些解码器会输出它们的解码结果 。另外，就像我们对编码器的输入所做的那样，我们会嵌入并添加位置编码给那些解码器，来表示每个单词的位置。

而那些解码器中的自注意力层表现的模式与编码器不同：在解码器中，自注意力层只被允许处理输出序列中更靠前的那些位置。在softmax步骤前，它会把后面的位置给隐去（把它们设为-inf）。

这个“编码-解码注意力层”工作方式基本就像多头自注意力层一样，只不过它是通过在它下面的层来创造查询矩阵，并且从编码器的输出中取得键/值矩阵。

## **Attention**

我在以前的文章中讲过，Attention 如果用一句话来描述，那就是 encoder 层的输出经过加权平均后再输入到 decoder 层中。它主要应用在 seq2seq 模型中，这个加权可以用矩阵来表示，也叫 Attention 矩阵。它表示对于某个时刻的输出 y，它在输入 x 上各个部分的注意力。这个注意力就是我们刚才说到的加权。

Attention 又分为很多种，其中两种比较典型的有加性 Attention 和乘性 Attention。加性 Attention 对于输入的隐状态 ![[公式]](https://www.zhihu.com/equation?tex=h_t) 和输出的隐状态 ![[公式]](https://www.zhihu.com/equation?tex=s_t) 直接做 concat 操作，得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Bs_t%3B+h_t%5D) ，乘性 Attention 则是对输入和输出做 dot 操作。

在 Google 这篇论文中，使用的 Attention 模型是乘性 Attention。

我在之前讲 [ESIM](https://zhuanlan.zhihu.com/p/47580077) 模型的文章里面写过一个 soft-align-attention，大家可以参考体会一下。

## **Self-Attention**

![img](https://pic2.zhimg.com/80/v2-fbb5dbc286b9f9cec2ddbc5eae2bf5a9_hd.jpg)

上面我们说attention机制的时候，都会说到两个隐状态，分别是 ![[公式]](https://www.zhihu.com/equation?tex=h_i) 和 ![[公式]](https://www.zhihu.com/equation?tex=s_t)。前者是输入序列第 i个位置产生的隐状态，后者是输出序列在第 t 个位置产生的隐状态。所谓 self-attention 实际上就是，输出序列就是输入序列。因而自己计算自己的 attention 得分。

从编码器输入的句子首先会经过一个自注意力（self-attention）层，这层帮助编码器在对每个单词编码时关注输入句子的其他单词。

自注意力层的输出会传递到前馈（feed-forward）神经网络中。每个位置的单词对应的前馈神经网络都完全一样（译注：另一种解读就是一层窗口为一个单词的一维卷积神经网络）。

### 从宏观视角看自注意力机制

例如，下列句子是我们想要翻译的输入句子：

> The animal didn't cross the street because it was too tired

这个“it”在这个句子是指什么呢？它指的是street还是这个animal呢？这对于人类来说是一个简单的问题，但是对于算法则不是。

当模型处理这个单词“it”的时候，自注意力机制会允许“it”与“animal”建立联系。

随着模型处理输入序列的每个单词，自注意力会关注整个输入序列的所有单词，帮助模型对本单词更好地进行编码。

如果你熟悉RNN（循环神经网络），回忆一下它是如何维持隐藏层的。RNN会将它已经处理过的前面的所有单词/向量的表示与它正在处理的当前单词/向量结合起来。而自注意力机制会将所有相关单词的理解融入到我们正在处理的单词中。

### **从微观视角看自注意力机制**

首先我们了解一下如何使用向量来计算自注意力，然后来看它实怎样用矩阵来实现。

计算自注意力的第一步就是从每个编码器的输入向量（每个单词的词向量）中生成三个向量。也就是说对于每个单词，我们创造一个查询向量、一个键向量和一个值向量。这三个向量是通过词嵌入与三个权重矩阵后相乘创建的。

可以发现这些新向量在维度上比词嵌入向量更低。他们的维度是64，而词嵌入和编码器的输入/输出向量的维度是512. 但实际上不强求维度更小，这只是一种基于架构上的选择，它可以使多头注意力（multiheaded attention）的大部分计算保持不变。



![img](https://pic2.zhimg.com/80/v2-bac717483cbeb04d1b5ef393eb87a16d_hd.jpg)

X1与WQ权重矩阵相乘得到q1, 就是与这个单词相关的查询向量。最终使得输入序列的每个单词的创建一个查询向量、一个键向量和一个值向量。

**什么是查询向量、键向量和值向量向量？**

它们都是有助于计算和理解注意力机制的抽象概念。请继续阅读下文的内容，你就会知道每个向量在计算注意力机制中到底扮演什么样的角色。

计算自注意力的第二步是计算得分。假设我们在为这个例子中的第一个词“Thinking”计算自注意力向量，我们需要拿输入句子中的每个单词对“Thinking”打分。这些分数决定了在编码单词“Thinking”的过程中有多重视句子的其它部分。

这些分数是通过打分单词（所有输入句子的单词）的键向量与“Thinking”的查询向量相点积来计算的。所以如果我们是处理位置最靠前的词的自注意力的话，第一个分数是q1和k1的点积，第二个分数是q1和k2的点积。

![img](https://pic2.zhimg.com/80/v2-373fb39e650fa85976bbb6eaf67b31ed_hd.jpg)

第三步和第四步是将分数除以8(8是论文中使用的键向量的维数64的平方根，这会让梯度更稳定。这里也可以使用其它值，8只是默认值)，然后通过softmax传递结果。softmax的作用是使所有单词的分数归一化，得到的分数都是正值且和为1。

![img](https://pic2.zhimg.com/80/v2-5591c2b55d9e31e744f884bacf959b45_hd.jpg)

这个softmax分数决定了每个单词对编码当下位置（“Thinking”）的贡献。显然，已经在这个位置上的单词将获得最高的softmax分数，但有时关注另一个与当前单词相关的单词也会有帮助。

<font color=blue>第五步是将每个值向量乘以softmax分数(这是为了准备之后将它们求和)。这里的直觉是希望关注语义上相关的单词，并弱化不相关的单词(例如，让它们乘以0.001这样的小数)。</font>

第六步是对加权值向量求和（译注：自注意力的另一种解释就是在编码某个单词时，就是将所有单词的表示（值向量）进行加权求和，而权重是通过该词的表示（键向量）与被编码词表示（查询向量）的点积并通过softmax得到。），然后即得到自注意力层在该位置的输出(在我们的例子中是对于第一个单词)。

第六步是对加权值向量求和（译注：自注意力的另一种解释就是在编码某个单词时，就是将所有单词的表示（值向量）进行加权求和，而权重是通过该词的表示（键向量）与被编码词表示（查询向量）的点积并通过softmax得到。），然后即得到自注意力层在该位置的输出(在我们的例子中是对于第一个单词)。

![img](https://pic4.zhimg.com/80/v2-609de8f8f8e628e6a9ca918230c70d67_hd.jpg)

这样自自注意力的计算就完成了。得到的向量就可以传给前馈神经网络。然而实际中，这些计算是以矩阵形式完成的，以便算得更快。那我们接下来就看看如何用矩阵实现的。

### **通过矩阵运算实现自注意力机制**

第一步是计算查询矩阵、键矩阵和值矩阵。为此，我们将将输入句子的词嵌入装进矩阵X中，将其乘以我们训练的权重矩阵(WQ，WK，WV)。

![](https://cdn.jsdelivr.net/gh/donttal/figurebed/img/计算查询矩阵.jpeg)

x矩阵中的每一行对应于输入句子中的一个单词。我们再次看到词嵌入向量 (512，或图中的4个格子)和q/k/v向量(64，或图中的3个格子)的大小差异。

最后，由于我们处理的是矩阵，我们可以将步骤2到步骤6合并为一个公式来计算自注意力层的输出。

![](https://cdn.jsdelivr.net/gh/donttal/figurebed/img/自注意力层输出.jpeg)

## Context-Attention

context-attention 是 encoder 和 decoder 之间的 attention，是两个不同序列之间的attention，与来源于自身的 self-attention 相区别。

不管是哪种 attention，我们在计算 attention 权重的时候，可以选择很多方式，常用的方法有

- additive attention
- local-base
- general
- dot-product
- scaled dot-product

Transformer模型采用的是最后一种：scaled dot-product attention。

## **Scaled Dot-Product Attention**

那么什么是 scaled dot-product attention 呢？

Google 在论文中对 Attention 机制这么来描述：

> An attention function can be described as a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, **where the weight assigned to each value is computed by a compatibility of the query with the corresponding key.**

通过 query 和 key 的相似性程度来确定 value 的权重分布。论文中的公式长下面这个样子：


![[公式]](https://www.zhihu.com/equation?tex=Attention%28Q%2CK%2CV%29%3Dsoftmax%28%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%29V)

看到 Q，K，V 会不会有点晕，没事，后面会解释。

scaled dot-product attention 和 dot-product attention 唯一的区别就是，scaled dot-product attention 有一个缩放因子， 叫![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B%5Csqrt%7Bd_k%7D%7D) 。 ![[公式]](https://www.zhihu.com/equation?tex=d_k) 表示 Key 的维度，默认用 64。

论文里对于 ![[公式]](https://www.zhihu.com/equation?tex=d_k) 的作用这么来解释：对于 ![[公式]](https://www.zhihu.com/equation?tex=d_k) 很大的时候，点积得到的结果维度很大，使得结果处于softmax函数梯度很小的区域。这时候除以一个缩放因子，可以一定程度上减缓这种情况。

scaled dot-product attention 的结构图如下所示。

![img](https://pic3.zhimg.com/80/v2-e2bc3a470b359e8bdce750843140897e_hd.jpg)

现在来说下 K、Q、V 分别代表什么：

- 在 encoder 的 self-attention 中，Q、K、V 都来自同一个地方，它们是上一层 encoder 的输出。对于第一层 encoder，它们就是 word embedding 和 positional encoding 相加得到的输入。
- 在 decoder 的 self-attention 中，Q、K、V 也是自于同一个地方，它们是上一层 decoder 的输出。对于第一层 decoder，同样也是 word embedding 和 positional encoding 相加得到的输入。但是对于 decoder，我们不希望它能获得下一个 time step (即将来的信息，不想让他看到它要预测的信息)，因此我们需要进行 sequence masking。
- 在 encoder-decoder attention 中，Q 来自于 decoder 的上一层的输出，K 和 V 来自于 encoder 的输出，K 和 V 是一样的。
- Q、K、V 的维度都是一样的，分别用 ![[公式]](https://www.zhihu.com/equation?tex=d_Q) 、![[公式]](https://www.zhihu.com/equation?tex=d_K) 和 ![[公式]](https://www.zhihu.com/equation?tex=d_V) 来表示

目前可能描述有有点抽象，不容易理解。结合一些应用来说，比如，如果是在自动问答任务中的话，Q 可以代表答案的词向量序列，取 K = V 为问题的词向量序列，那么输出就是所谓的 Aligned Question Embedding。

Google 论文的主要贡献之一是它表明了内部注意力在机器翻译 (甚至是一般的Seq2Seq任务）的序列编码上是相当重要的，而之前关于 Seq2Seq 的研究基本都只是把注意力机制用在解码端。

## **Scaled Dot-Product Attention 实现**

```python
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播.
        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            # 给需要 mask 的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
	# 计算softmax
        attention = self.softmax(attention)
	# 添加dropout
        attention = self.dropout(attention)
	# 和V做点积
        context = torch.bmm(attention, v)
        return context, attention
```

## **Multi-head attention**

理解了 Scaled dot-product attention，Multi-head attention 也很容易理解啦。论文提到，他们发现将 Q、K、V 通过一个线性映射之后，分成 h 份，对每一份进行 scaled dot-product attention 效果更好。然后，把各个部分的结果合并起来，再次经过线性映射，得到最终的输出。这就是所谓的 multi-head attention。上面的超参数 h 就是 heads 的数量。论文默认是 8。

multi-head attention 的结构图如下所示。

![img](https://pic4.zhimg.com/80/v2-b8b4befd9b96318047ffb3251421cbe3_hd.jpg)

值得注意的是，上面所说的分成 h 份是在 ![[公式]](https://www.zhihu.com/equation?tex=d_Q) 、![[公式]](https://www.zhihu.com/equation?tex=d_K) 和 ![[公式]](https://www.zhihu.com/equation?tex=d_V)的维度上进行切分。因此进入到scaled dot-product attention 的 ![[公式]](https://www.zhihu.com/equation?tex=d_K) 实际上等于未进入之前的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7BD_K%7D%7Bh%7D) 。

Multi-head attention 的公式如下：

![[公式]](https://www.zhihu.com/equation?tex=MultiHead%28Q%2C+K%2C+V%29+%3D+Concat%28head_1%2C+...%2C+head_h%29W%5E0)

其中，

![[公式]](https://www.zhihu.com/equation?tex=head_i+%3D+Attention%28QW_i%5EQ%2C+KW_i%5EK%2C+VW_i%5EV%29)

在论文里面， ![[公式]](https://www.zhihu.com/equation?tex=d_%7Bmodel%7D) = 512，h = 8，所以在 scaled dot-product attention 里面的

![[公式]](https://www.zhihu.com/equation?tex=d_Q+%3D+d_K+%3D+d_V+%3D+d_%7Bmodel%7D+%2F+h+%3D+512+%2F+8+%3D+64+) 。

可以看出，所谓 Multi-Head，就是只多做几次同样的事情，同时参数不共享，然后把结果拼接。



通过增加一种叫做“多头”注意力（“multi-headed” attention）的机制，论文进一步完善了自注意力层，并在两方面提高了注意力层的性能：

1.它扩展了模型专注于不同位置的能力。在上面的例子中，虽然每个编码都在z1中有或多或少的体现，但是它可能被实际的单词本身所支配。如果我们翻译一个句子，比如“The animal didn’t cross the street because it was too tired”，我们会想知道“it”指的是哪个词，这时模型的“多头”注意机制会起到作用。

2.它给出了注意力层的多个“表示子空间”（representation subspaces）。接下来我们将看到，对于“多头”注意机制，我们有多个查询/键/值权重矩阵集(Transformer使用八个注意力头，因此我们对于每个编码器/解码器有八个矩阵集合)。这些集合中的每一个都是随机初始化的，在训练之后，每个集合都被用来将输入词嵌入(或来自较低编码器/解码器的向量)投影到不同的表示子空间中。





![img](https://pic2.zhimg.com/80/v2-6214664f4f8d080ab41e6909f15b1f69_hd.jpg)



在“多头”注意机制下，我们为每个头保持独立的查询/键/值权重矩阵，从而产生不同的查询/键/值矩阵。和之前一样，我们拿X乘以WQ/WK/WV矩阵来产生查询/键/值矩阵。

如果我们做与上述相同的自注意力计算，只需八次不同的权重矩阵运算，我们就会得到八个不同的Z矩阵。





![img](https://pic4.zhimg.com/80/v2-9adff6a99688705389e3d96bc0a72d2f_hd.jpg)





这给我们带来了一点挑战。前馈层不需要8个矩阵，它只需要一个矩阵(由每一个单词的表示向量组成)。所以我们需要一种方法把这八个矩阵压缩成一个矩阵。那该怎么做？其实可以直接把这些矩阵拼接在一起，然后用一个附加的权重矩阵WO与它们相乘。



![img](https://pic2.zhimg.com/80/v2-745f36fdfed78f28629bbdd4bac8660d_hd.jpg)





这几乎就是多头自注意力的全部。这确实有好多矩阵，我们试着把它们集中在一个图片中，这样可以一眼看清。

![img](https://pic1.zhimg.com/80/v2-eecbb4c1d2575844f9fb37cc144bc94c_hd.jpg)



既然我们已经摸到了注意力机制的这么多“头”，那么让我们重温之前的例子，看看我们在例句中编码“it”一词时，不同的注意力“头”集中在哪里：



![img](https://pic3.zhimg.com/80/v2-f268e009970206d27c6762fd634c2d12_hd.jpg)



当我们编码“it”一词时，一个注意力头集中在“animal”上，而另一个则集中在“tired”上，从某种意义上说，模型对“it”一词的表达在某种程度上是“animal”和“tired”的代表。

然而，如果我们把所有的attention都加到图示里，事情就更难解释了：





![img](https://pic4.zhimg.com/80/v2-fc5f733a4f5dfc654f1211ba25eb4797_hd.jpg)

## **Multi-head attention 实现**

```python
class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, 15927210044_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // 15927210044_heads
        self.15927210044_heads = 15927210044_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * 15927210044_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * 15927210044_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * 15927210044_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
	
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
	# 残差连接
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        # scaled dot product attention
        scale = (key.size(-1)) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention
```



上面代码中出现的 [Residual connection](https://zhuanlan.zhihu.com/p/47846504) 我在之前一篇文章中讲过，这里不再赘述，只解释 Layer normalization。

## **Layer normalization**

> Normalization 有很多种，但是它们都有一个共同的目的，那就是把输入转化成均值为 0 方差为 1 的数据。我们在把数据送入激活函数之前进行 normalization（归一化），因为我们不希望输入数据落在激活函数的饱和区。

说到 normalization，那就肯定得提到 Batch Normalization。

BN 的主要思想就是：在每一层的每一批数据上进行归一化。我们可能会对输入数据进行归一化，但是经过该网络层的作用后，我们的数据已经不再是归一化的了。随着这种情况的发展，数据的偏差越来越大，我的反向传播需要考虑到这些大的偏差，这就迫使我们只能使用较小的学习率来防止梯度消失或者梯度爆炸。

BN 的具体做法就是对每一小批数据，在批这个方向上做归一化。如下图所示：

![img](https://pic2.zhimg.com/80/v2-c830c74003737b7198a2800015b550d9_hd.jpg)

可以看到，右半边求均值是**沿着数据 batch N 的方向进行的**！

Batch normalization 的计算公式如下：

![[公式]](https://www.zhihu.com/equation?tex=BN%28x_i%29+%3D+%5Calpha%5Ctimes%5Cfrac%7Bx_i+-+%5Cmu_b%7D%7B%5Csqrt%7B%5Csigma_B%5E2+%2B+%5Cepsilon%7D%7D+%2B+%5Cbeta)

那么什么是 Layer normalization 呢？它也是归一化数据的一种方式，不过 LN 是**在每一个样本上计算均值和方差，而不是 BN 那种在批方向计算均值和方差**！

下面是 LN 的示意图：

![img](https://pic1.zhimg.com/80/v2-71b795c20bec7ef43397e96d1309e9d4_hd.jpg)

和上面的 BN 示意图一比较就可以看出二者的区别啦！

下面看一下 LN 的公式：

![[公式]](https://www.zhihu.com/equation?tex=LN%28x_i%29+%3D+%5Calpha%5Ctimes%5Cfrac%7Bx_i+-+%5Cmu_L%7D%7B%5Csqrt%7B%5Csigma_L%5E2+%2B+%5Cepsilon%7D%7D+%2B+%5Cbeta)

## **Mask**

mask 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果。Transformer 模型里面涉及两种 mask，分别是 padding mask 和 sequence mask。

其中，padding mask 在所有的 scaled dot-product attention 里面都需要用到，而 sequence mask 只有在 decoder 的 self-attention 里面用到。

**Padding Mask**

什么是 padding mask 呢？因为每个批次输入序列长度是不一样的也就是说，我们要对输入序列进行对齐。具体来说，就是给在较短的序列后面填充 0。因为这些填充的位置，其实是没什么意义的，所以我们的 attention 机制不应该把注意力放在这些位置上，所以我们需要进行一些处理。

具体的做法是，把这些位置的值加上一个非常大的负数(负无穷)，这样的话，经过 softmax，这些位置的概率就会接近0！

而我们的 padding mask 实际上是一个张量，每个值都是一个 Boolean，值为 false 的地方就是我们要进行处理的地方。

实现：

```python
def padding_mask(seq_k, seq_q):
    # seq_k 和 seq_q 的形状都是 [B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask
```

**Sequence mask**

文章前面也提到，sequence mask 是为了使得 decoder 不能看见未来的信息。也就是对于一个序列，在 time_step 为 t 的时刻，我们的解码输出应该只能依赖于 t 时刻之前的输出，而不能依赖 t 之后的输出。因此我们需要想一个办法，把 t 之后的信息给隐藏起来。

那么具体怎么做呢？也很简单：**产生一个上三角矩阵，上三角的值全为 1，下三角的值权威0，对角线也是 0**。把这个矩阵作用在每一个序列上，就可以达到我们的目的啦。

具体的代码实现如下：

```python
def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask
```



效果如下，

![img](https://pic1.zhimg.com/80/v2-4ae5ac1ce585b382b32b023a8aa830f0_hd.jpg)

- 对于 decoder 的 self-attention，里面使用到的 scaled dot-product attention，同时需要padding mask 和 sequence mask 作为 attn_mask，具体实现就是两个 mask 相加作为attn_mask。
- 其他情况，attn_mask 一律等于 padding mask。

## **Positional Embedding**

![img](https://pic4.zhimg.com/80/v2-ac9c8d239071146e24db823e2ae4f43b_hd.jpg)

现在的 Transformer 架构还没有提取序列顺序的信息，这个信息对于序列而言非常重要，如果缺失了这个信息，可能我们的结果就是：所有词语都对了，但是无法组成有意义的语句。

为了解决这个问题。论文使用了 Positional Embedding：对序列中的词语出现的位置进行编码。

在实现的时候使用正余弦函数。公式如下：

![[公式]](https://www.zhihu.com/equation?tex=PE%28pos%2C+2i%29+%3D+sin%28pos%2F10000%5E%7B2i%2Fd_%7Bmodel%7D%7D%29)

![[公式]](https://www.zhihu.com/equation?tex=PE%28pos%2C+2i%2B1%29+%3D+cos%28pos%2F10000%5E%7B2i%2Fd_%7Bmodel%7D%7D%29)

其中，pos 是指词语在序列中的位置。可以看出，在**偶数位置，使用正弦编码，在奇数位置，使用余弦编码**。

从编码公式中可以看出，给定词语的 pos，我们可以把它编码成一个 ![[公式]](https://www.zhihu.com/equation?tex=d_%7Bmodel%7D) 的向量。也就是说，位置编码的每一个维度对应正弦曲线，波长构成了从 ![[公式]](https://www.zhihu.com/equation?tex=2%5Cpi) 到 ![[公式]](https://www.zhihu.com/equation?tex=10000%5Ctimes2%5Cpi) 的等比数列。

上面的位置编码是**绝对位置编码**。但是词语的**相对位置**也非常重要。这就是论文为什么要使用三角函数的原因！

正弦函数能够表达相对位置信息，主要数学依据是以下两个公式：

![[公式]](https://www.zhihu.com/equation?tex=sin%28%5Calpha+%2B+%5Cbeta%29+%3D+sin%5Calpha+cos%5Cbeta+%2B+cos%5Calpha+sin+%5Cbeta+%5C%5C+cos%28%5Calpha+%2B+%5Cbeta%29+%3D+cos%5Calpha+cos%5Cbeta+-+sin%5Calpha+sin+%5Cbeta)

上面的公式说明，对于词汇之间的位置偏移 k， ![[公式]](https://www.zhihu.com/equation?tex=PE%28pos+%2B+k%29) 可以表示成 ![[公式]](https://www.zhihu.com/equation?tex=PE%28pos%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=PE%28k%29)组合的形式，相当于有了可以表达相对位置的能力。

具体实现如下：

```python
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_len):
        """初始化。
        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
          [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
          for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))
        
        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)
    def forward(self, input_len):
        """神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """
        
        # 找出这一批序列的最大长度
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor(
          [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)
```

## **Position-wise Feed-Forward network**

这是一个全连接网络，包含两个线性变换和一个非线性函数(实际上就是 ReLU)。公式如下

![[公式]](https://www.zhihu.com/equation?tex=FFN+%3D+max%280%2C++xW_1+%2B+b_1%29W_2+%2B+b_2)

这个线性变换在不同的位置都表现地一样，并且在不同的层之间使用不同的参数。

**这里实现上用到了两个一维卷积。**

实现如下:

```python
class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output
```

## **最终的线性变换和Softmax层**

解码组件最后会输出一个实数向量。我们如何把浮点数变成一个单词？这便是线性变换层要做的工作，它之后就是Softmax层。

线性变换层是一个简单的全连接神经网络，它可以把解码组件产生的向量投射到一个比它大得多的、被称作对数几率（logits）的向量里。

不妨假设我们的模型从训练集中学习一万个不同的英语单词（我们模型的“输出词表”）。因此对数几率向量为一万个单元格长度的向量——每个单元格对应某一个单词的分数。

接下来的Softmax 层便会把那些分数变成概率（都为正数、上限1.0）。概率最高的单元格被选中，并且它对应的单词被作为这个时间步的输出。

## Transformer的实现

现在可以开始完成 Transformer 模型的构建了，encoder 端和 decoder 端分别都有 6 层，实现如下，首先是 encoder 端，

```python
class EncoderLayer(nn.Module):
	"""Encoder的一层。"""

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs, padding_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
	"""多层EncoderLayer组成Encoder。"""

    def __init__(self,
               vocab_size,
               max_seq_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_mask = padding_mask(inputs, inputs)

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions

```



然后是 Decoder 端，

```python
class DecoderLayer(nn.Module):

    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self,
              dec_inputs,
              enc_outputs,
              self_attn_mask=None,
              context_attn_mask=None):
        # self attention, all inputs are decoder inputs
        dec_output, self_attention = self.attention(
          dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        # context attention
        # query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.attention(
          enc_outputs, enc_outputs, dec_output, context_attn_mask)

        # decoder's output, or context
        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention


class Decoder(nn.Module):

    def __init__(self,
               vocab_size,
               max_seq_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):
        super(Decoder, self).__init__()

        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
          [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_padding_mask = padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs)
        self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(
            output, enc_output, self_attn_mask, context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions
```



组合一下，就是 Transformer 模型。

```python
class Transformer(nn.Module):

    def __init__(self,
               src_vocab_size,
               src_max_len,
               tgt_vocab_size,
               tgt_max_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.2):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)

        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        context_attn_mask = padding_mask(tgt_seq, src_seq)

        output, enc_self_attn = self.encoder(src_seq, src_len)

        output, dec_self_attn, ctx_attn = self.decoder(
          tgt_seq, tgt_len, output, context_attn_mask)

        output = self.linear(output)
        output = self.softmax(output)

        return output, enc_self_attn, dec_self_attn, ctx_attn
```

## 项目地址

Github 地址: [pengshuang/Transformer](https://github.com/pengshuang/Transformer)

## 训练部分总结

既然我们已经过了一遍完整的transformer的前向传播过程，那我们就可以直观感受一下它的训练过程。

在训练过程中，一个未经训练的模型会通过一个完全一样的前向传播。但因为我们用有标记的训练集来训练它，所以我们可以用它的输出去与真实的输出做比较。

为了把这个流程可视化，不妨假设我们的输出词汇仅仅包含六个单词：“a”, “am”, “i”, “thanks”, “student”以及 “<eos>”（end of sentence的缩写形式）。



![img](https://pic3.zhimg.com/80/v2-a90ee85468dfcd0a2db49bcc6319fc1a_hd.jpg)



我们模型的输出词表在我们训练之前的预处理流程中就被设定好。



一旦我们定义了我们的输出词表，我们可以使用一个相同宽度的向量来表示我们词汇表中的每一个单词。这也被认为是一个one-hot 编码。所以，我们可以用下面这个向量来表示单词“am”：





![img](https://pic2.zhimg.com/80/v2-fb499dc05f731b123f2aa395b81ae5e5_hd.jpg)



例子：对我们输出词表的one-hot 编码



接下来我们讨论模型的损失函数——这是我们用来在训练过程中优化的标准。通过它可以训练得到一个结果尽量准确的模型。



**损失函数**



比如说我们正在训练模型，现在是第一步，一个简单的例子——把“merci”翻译为“thanks”。



这意味着我们想要一个表示单词“thanks”概率分布的输出。但是因为这个模型还没被训练好，所以不太可能现在就出现这个结果。





![img](https://pic1.zhimg.com/80/v2-65f30ddcbe1d8fde584a2141bbc191c0_hd.jpg)



因为模型的参数（权重）都被随机的生成，（未经训练的）模型产生的概率分布在每个单元格/单词里都赋予了随机的数值。我们可以用真实的输出来比较它，然后用反向传播算法来略微调整所有模型的权重，生成更接近结果的输出。



你会如何比较两个概率分布呢？我们可以简单地用其中一个减去另一个。更多细节请参考交叉熵和KL散度。



交叉熵：

https://colah.github.io/posts/2015-09-Visual-Information/

KL散度：

https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained



但注意到这是一个过于简化的例子。更现实的情况是处理一个句子。例如，输入“je suis étudiant”并期望输出是“i am a student”。那我们就希望我们的模型能够成功地在这些情况下输出概率分布：

- 每个概率分布被一个以词表大小（我们的例子里是6，但现实情况通常是3000或10000）为宽度的向量所代表。
- 第一个概率分布在与“i”关联的单元格有最高的概率
- 第二个概率分布在与“am”关联的单元格有最高的概率
- 以此类推，第五个输出的分布表示“<end of sentence>”关联的单元格有最高的概率



![img](https://pic4.zhimg.com/80/v2-4fde20f7e67ae981ff478a6b8219d557_hd.jpg)



依据例子训练模型得到的目标概率分布。



在一个足够大的数据集上充分训练后，我们希望模型输出的概率分布看起来像这个样子：





![img](https://pic3.zhimg.com/80/v2-dcbcf7afeefd60bb5533a0db99dbbdf6_hd.jpg)



我们期望训练过后，模型会输出正确的翻译。当然如果这段话完全来自训练集，它并不是一个很好的评估指标（参考：交叉验证，链接https://www.youtube.com/watch?v=TIgfjmp-4BA）。注意到每个位置（词）都得到了一点概率，即使它不太可能成为那个时间步的输出——这是softmax的一个很有用的性质，它可以帮助模型训练。



因为这个模型一次只产生一个输出，不妨假设这个模型只选择概率最高的单词，并把剩下的词抛弃。这是其中一种方法（叫贪心解码）。另一个完成这个任务的方法是留住概率最靠高的两个单词（例如I和a），那么在下一步里，跑模型两次：其中一次假设第一个位置输出是单词“I”，而另一次假设第一个位置输出是单词“me”，并且无论哪个版本产生更少的误差，都保留概率最高的两个翻译结果。然后我们为第二和第三个位置重复这一步骤。这个方法被称作集束搜索（beam search）。在我们的例子中，集束宽度是2（因为保留了2个集束的结果，如第一和第二个位置），并且最终也返回两个集束的结果（top_beams也是2）。这些都是可以提前设定的参数。



**再进一步**



我希望通过上文已经让你们了解到Transformer的主要概念了。如果你想在这个领域深入，我建议可以走以下几步：阅读Attention Is All You Need，Transformer博客和Tensor2Tensor announcement，以及看看Łukasz Kaiser的介绍，了解模型和细节。



Attention Is All You Need：

https://arxiv.org/abs/1706.03762

Transformer博客：

https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html

Tensor2Tensor announcement：

https://ai.googleblog.com/2017/06/accelerating-deep-learning-research.html

Łukasz Kaiser的介绍：

https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb



接下来可以研究的工作：

- Depthwise Separable Convolutions for Neural Machine Translation
  https://arxiv.org/abs/1706.03059
- One Model To Learn Them All
  https://arxiv.org/abs/1706.05137
- Discrete Autoencoders for Sequence Models
  https://arxiv.org/abs/1801.09797
- Generating Wikipedia by Summarizing Long Sequences
  https://arxiv.org/abs/1801.10198
- Image Transformer
  https://arxiv.org/abs/1802.05751
- Training Tips for the Transformer Model
  https://arxiv.org/abs/1804.00247
- Self-Attention with Relative Position Representations
  https://arxiv.org/abs/1803.02155
- Fast Decoding in Sequence Models using Discrete Latent Variables
  https://arxiv.org/abs/1803.03382
- Adafactor: Adaptive Learning Rates with Sublinear Memory Cost
  https://arxiv.org/abs/1804.04235