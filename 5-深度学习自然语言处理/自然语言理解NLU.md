# 自然语言理解NLU

自然语言理解就是希望机器像人一样，具备正常人的语言理解能力，由于自然语言在理解上有很多难点(下面详细说明)，所以 NLU 是至今还远不如人类的表现。

## 意图识别和实体提取

自然语言理解让机器从各种自然语言的表达中，区分出来，哪些话归属于这个意图；而那些表达不是归于这一类的，而不再依赖那么死板的关键词。

比如经过训练后，机器能够识别“帮我推荐一家附近的餐厅”，就不属于“订机票”这个意图的表达。

通过训练，机器还能够在“看看航班，下周二出发去纽约”句子当中自动提取出来“纽约”，这两个字指的是目的地这个概念（即实体）；“下周二”指的是出发时间。

## 自然语言理解（NLU）的难点

下面先列举一些机器不容易理解的案例：

1. 校长说衣服上除了校徽别别别的
2. 过几天天天天气不好

**自然语言理解的5个难点：**

1. 语言的多样性
2. 语言的歧义性
3. 语言的鲁棒性[语音识别获得的文本，多字少字错字噪音]
4. 语言的知识依赖
5. 语言的上下文

## NLU 的实现方式

自然语言理解跟整个人工智能的发展历史类似，一共经历了3次迭代：

1. 基于规则的方法。
2. 基于统计的方法
3. 基于深度学习的方法

最早大家通过总结规律来判断自然语言的意图，常见的方法有：CFG、JSGF等。

后来出现了基于统计学的 NLU 方式，常见的方法有：[SVM](https://easyai.tech/ai-definition/svm/)、ME等。

随着深度学习的爆发，[CNN](https://easyai.tech/ai-definition/cnn/)、[RNN](https://easyai.tech/ai-definition/rnn/)、[LSTM](https://easyai.tech/ai-definition/lstm/) 都成为了最新的”统治者”。

到了2019年，[BERT](https://easyai.tech/ai-definition/bert/) 和 GPT-2 的表现震惊了业界，他们都是用了 [Transformer](https://easyai.tech/ai-definition/transformer/)，下面将重点介绍 Transformer，因为他是目前「最先进」的方法。

### **Transformer 和 CNN / RNN 的比较**

#### **语义特征提取能力**

**Transformer>>原生CNN=原生RNN**

从语义特征提取能力来说，目前实验支持如下结论：Transformer在这方面的能力非常显著地超过RNN和CNN（在考察语义类能力的任务WSD中，Transformer超过RNN和CNN大约4-8个绝对百分点），RNN和CNN两者能力差不太多。

#### **长距离特征捕获能力**

**Transformer>原生RNN>原生CNN**

原生CNN特征抽取器在这方面极为显著地弱于RNN和Transformer，Transformer微弱优于RNN模型(尤其在主语谓语距离小于13时)，能力由强到弱排序为Transformer>RNN>>CNN; 但在比较远的距离上（主语谓语距离大于13），RNN微弱优于Transformer，所以综合看，可以认为Transformer和RNN在这方面能力差不太多，而CNN则显著弱于前两者。

#### **任务综合特征抽取能力**

**Transformer>>原生CNN=原生CNN**

Transformer综合能力要明显强于RNN和CNN（你要知道，技术发展到现在阶段，BLEU绝对值提升1个点是很难的事情），而RNN和CNN看上去表现基本相当，貌似CNN表现略好一些。

#### **并行计算能力及运算效率**

Transformer Base最快，CNN次之，再次Transformer Big，最慢的是RNN。RNN比前两者慢了3倍到几十倍之间。