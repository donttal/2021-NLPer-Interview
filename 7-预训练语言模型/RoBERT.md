# NLP数据增强[网址](https://zhuanlan.zhihu.com/p/145521255)
[toc]

## 1.词汇替换
这种方法试图在不改变句子主旨的情况下替换文本中的单词。

### 1.1 基于词典的替换
在这种技术中，我们从句子中随机取出一个单词，并使用同义词词典将其替换为同义词。例如，我们可以使用WordNet的英语词汇数据库来查找同义词，然后执行替换。它是一个手动管理的数据库，其中包含单词之间的关系。

### 1.2 基于词向量的替换
在这种方法中，我们采用预先训练好的单词嵌入，如Word2Vec、GloVe、FastText、Sent2Vec，并使用嵌入空间中最近的相邻单词替换句子中的某些单词。Jiao et al.在他们的论文“TinyBert”中使用了这种技术，以提高他们的语言模型在下游任务上的泛化能力。Wang et al.使用它来增加学习主题模型所需的tweet。

例如，你可以用三个最相似的单词来替换句子中的单词，并得到文本的三个变体。
![](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvoKJVibWQTjU7GicApVmoRHn3ibMMlk5v30iaImpOH8lorLTz5oNNWx5WxuD8TkPC6oUemZ1Fl8AaMjTQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

 
###  1.3 Masked Language Model
像BERT、ROBERTA和ALBERT这样的Transformer模型已经接受了大量的文本训练，使用一种称为“Masked Language Modeling”的预训练，即模型必须根据上下文来预测遮盖的词汇。这可以用来扩充一些文本。例如，我们可以使用一个预训练的BERT模型并屏蔽文本的某些部分。然后，我们使用BERT模型来预测遮蔽掉的token。

因此，我们可以使用mask预测来生成文本的变体。与之前的方法相比，生成的文本在语法上更加连贯，因为模型在进行预测时考虑了上下文。

然而，这种方法的一个问题是，决定要屏蔽文本的哪一部分并不是一件小事。**你必须使用启发式的方法来决定掩码**，否则生成的文本将不保留原句的含义。

### 1.4 基于TF-IDF的词替换

这种增强方法是由Xie et al.在无监督数据增强论文中提出的。其基本思想是，TF-IDF分数较低的单词不能提供信息，因此可以在不影响句子的ground-truth的情况下替换它们。

## 2. 反向翻译
在这种方法中，我们利用机器翻译来解释文本，同时重新训练含义。Xie et al.使用这种方法来扩充未标注的文本，并在IMDB数据集中学习一个只有20个有标注样本的半监督模型。该方法优于之前的先进模型，该模型训练了25,000个有标注的样本。

反向翻译过程如下：

把一些句子(如英语)翻译成另一种语言，如法语

将法语句子翻译回英语句子。

检查新句子是否与原来的句子不同。如果是，那么我们使用这个新句子作为原始文本的数据增强。


## 3. 文本表面转换
这些是**使用正则表达式的简单的模式匹配**的转换，由Claude Coulombe在他的论文中介绍。

在本文中，他给出了一个将动词形式由简写转化为完整形式或者反过来的例子。我们可以通过这个来生成增强型文本。
![](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvoKJVibWQTjU7GicApVmoRHn3I5pxkuRPuXSBRRy43wbVZXP8F4xl3mHGpJ7bgIbz4t1lALiaYWTkjew/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

既然转换不应该改变句子的意思，我们可以看到，在扩展模棱两可的动词形式时，这可能会失败，比如：
![](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvoKJVibWQTjU7GicApVmoRHn3gbpUbibSXdLWicm7NGFDBTeYSsKpDk3TNIyj0bjN5fmeshugUyykeBibw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

为了解决这一问题，本文提出允许模糊收缩，但跳过模糊展开。
![](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvoKJVibWQTjU7GicApVmoRHn3iafKdaica9ZEibLgsNBcciaTaVpgPAaxZYZFnbYbKbJGvqmLe6Ria85ocaw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)


## 4. 随机噪声注入
这些方法的思想是在文本中加入噪声，使所训练的模型对扰动具有鲁棒性。

### 4.1 拼写错误注入

在这种方法中，我们在句子中的一些随机单词上添加拼写错误。这些拼写错误可以通过编程方式添加，也可以使用常见拼写错误的映射，如：https://github.com/makcedward/nlpaug/blob/master/model/spelling_en.txt。
![](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvoKJVibWQTjU7GicApVmoRHn38Zic6Bapa2dTS86aj09T0fVAH2FS63APQlhEQd4h2J5xrpRM06hicB9g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 4.2 QWERTY键盘错误注入

该方法试图模拟在QWERTY布局键盘上输入时发生的常见错误，这些错误是由于按键之间的距离非常近造成的。错误是根据键盘距离注入的。
![](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvoKJVibWQTjU7GicApVmoRHn3U4lrkvwh3IiaQFDH1AEiasSFTg7AiagZ5mRD5JAgCqlqM8B9YYXU4H4GA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 4.3 Unigram噪声

该方法已被Xie et al.和UDA论文所采用。其思想是用从单字符频率分布中采样的单词进行替换。这个频率基本上就是每个单词在训练语料库中出现的次数。
![](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvoKJVibWQTjU7GicApVmoRHn3vyhFGicbdFW7QsDwUic80zib8CwhdqJIRg6M1u23czxPribsUn9kYXYZFg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 4.4 Blank Noising

这个方法是由Xie et al.在他们的论文中提出的。其思想是用占位符标记替换一些随机单词。本文使用“_”作为占位符标记。在论文中，他们将其作为一种避免特定上下文过拟合的方法，以及语言模型的平滑机制。该技术有助于提高perplexity和BLEU评分。

![](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvoKJVibWQTjU7GicApVmoRHn3KGWyiamkViaM8cvZ6Znem4sIPf547dR671VlwatWmCNiaibBiaOAvGuDcbw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 4.5 句子打乱
这是一种朴素的技术，我们将训练文本中的句子打乱，以创建一个增强版本。

## 5. 实例交叉增强
这项技术是由Luque在他的关于TASS 2019情绪分析的论文中提出的。这项技术的灵感来自于遗传学中发生的染色体交叉操作。

该方法将tweets分为两部分，两个具有相同极性的随机推文(即正面/负面)进行交换。这个方法的假设是，即使结果是不符合语法和语义的，新文本仍将保留情感的极性。

![](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvoKJVibWQTjU7GicApVmoRHn3LTjjZmoS9cYUkHXLbFxZboTia66Goug4icHpd334HAia9wCYsesANSvrA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

这一技术对准确性没有影响，但有助于论文中极少数类的F1分数，如tweets较少的中性类。
![](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvoKJVibWQTjU7GicApVmoRHn3EkgRUabhS2MxaaUibiaMdibJzlwOwmrzIW0rr50HILhENcF6RxGAG8aHQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## 6. 语法树操作
这项技术已经在Coulombe的论文中使用。其思想是解析和生成原始句子的依赖关系树，使用规则对其进行转换，并生成改写后的句子。

例如，一个不改变句子意思的转换是句子从主动语态到被动语态的转换，反之亦然。

![](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvoKJVibWQTjU7GicApVmoRHn3IUSzKicMQjWYQkPI938fTjf04I8R5UEia0aa0Ucicko9jc911zm8dzwGQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 实现
要使用上述所有方法，可以使用名为nlpaug的python库：https://github.com/makcedward/nlpaug。它提供了一个简单且一致的API来应用这些技术。






