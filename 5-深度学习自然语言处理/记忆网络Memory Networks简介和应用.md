# 记忆网络Memory Networks简介和应用
[专栏链接](https://zhuanlan.zhihu.com/c_129532277)

## [Memory Networks](https://zhuanlan.zhihu.com/p/29590286)
FaceBook2014年发表的论文“Memory Networks”提出传统的深度学习模型（RNN、LSTM、GRU等）使用hidden states或者Attention机制作为他们的记忆功能，但是这种方法产生的记忆太小了，无法精确记录一段话中所表达的全部内容，也就是在将输入编码成dense vectors的时候丢失了很多信息。所以本文就提出了一种可读写的外部记忆模块，并将其和inference组件联合训练，最终得到一个可以被灵活操作的记忆模块。

简单来说，就是输入的文本经过Input模块编码成向量，然后将其作为Generalization模块的输入，该模块根据输入的向量对memory进行读写操作，即对记忆进行更新。然后Output模块会根据Question（也会进过Input模块进行编码）对memory的内容进行权重处理，将记忆按照与Question的相关程度进行组合得到输出向量，最终Response模块根据输出向量编码生成一个自然语言的答案出来。


## [End-To-End Memory Networks](https://zhuanlan.zhihu.com/p/29679742)
这是Facebook AI在Memory networks之后提出的一个更加完善的模型，前文中我们已经说到，其I和G模块并未进行复杂操作，只是将原始文本进行向量化并保存，没有对输入文本进行适当的修改就直接保存为memory。而O和R模块承担了主要的任务，但是从最终的目标函数可以看出，在O和R部分都需要监督，也就是我们需要知道O选择的相关记忆是否正确，R生成的答案是否正确。这就限制了模型的推广，太多的地方需要监督，不太容易使用反向传播进行训练。因此，本文提出了一种end-to-end的模型，可以视为一种continuous form的Memory Network，而且需要更少的监督数据便可以进行训练。论文中提出了单层和多层两种架构，多层其实就是将单层网络进行stack。
![](https://pic1.zhimg.com/80/v2-9da86e80a07e2d8d1b777055b4fcefac_hd.jpg)

模型主要的参数包括A,B,C,W四个矩阵，其中A,B,C三个矩阵就是embedding矩阵，主要是将输入文本和Question编码成词向量，W是最终的输出矩阵。从上图可以看出，对于输入的句子s分别会使用A和C进行编码得到Input和Output的记忆模块，Input用来跟Question编码得到的向量相乘得到每句话跟q的相关性，Output则与该相关性进行加权求和得到输出向量。然后再加上q并传入最终的输出层。

### Tensorflow实现
**bAbI QA建模：**这部分代码参考https://github.com/domluna/memn2n，先简单介绍一下数据集，bAbI是facebook提出的，里面包含了20种问题类型，分别代表了不同的推理形式。
**PTB 语言模型建模**:这部分的代码可以参考[https://github.com/carpedm20/MemN2N-tensorflow]，这就是一个很传统的语言建模任务，其实就是给定一个词序列预测下一个词出现的概率，数据集使用的是PTB（训练集、验证集、测试集分别包含929K，73K，82K个单词，vocab包含10000个单词）。

[参考网址](https://zhuanlan.zhihu.com/p/29679742)

## [open-domain QA 应用](https://zhuanlan.zhihu.com/p/29763127)
在只有Question的时候直接回答呢？这里需要引入知识库（Knowledge Bases， KBS）的概念，其实在Memory Networks兴起之前，传统的QA都是通过知识库进行检索或者信息抽取等方式进行建模的（这个有时间应该也会专门研究一下，这里先不进行引申）。所谓知识库，其实就是将互联网上的信息经过专家人工提取和构造，以三元组的形式存储下来（subject， relationship， object），是一种非常结构化的信息，比较知名的有FreeBase。有了知识库，那么给定一个问题我们自然可以进行检索记忆并作出回答。那么Memory Network的优点是什么呢相比传统方法，因为问题往往是以自然语言的方式提出，而知识库是高度结构化的组织，所以如何进行检索或者信息抽取其实是一个很困难的工作，所以就引入了深度学习==另外还有一个问题就是，前面所提到的Memory Network记忆的都是一个相对较小的文档存储起来，那么面对KB这么庞大的知识库应该如何解决内存容量问题呢？下面我们进行探索。

### Large-scale Simple Question Answering with Memory Networks
数据集：Simple Question Answering
1. 存储KB，第一步是使用Input组件解析知识库并将其存入Memory之中。
2. 训练，第二步是使用Simple QA数据集的所有QA对作为输入，经过Input、Output、Response三个组件进行预测并使用反向传播训练网络中的参数
3. 泛化，最后使用Reverb数据集直接接入模型（不在进行重新训练），使用Generalization模块将其与Memory中已经存在的facts关联，看最终效果如何

这里专门加入了Reverb数据集是因为作者认为传统的数据集很小不具有扩展性，而且在单一数据集上训练出的模型在别的数据集上的泛化能力可能会很差，为了证明本论文训练出来的模型具有这种泛化能力加入了这一项。Reverb数据集跟FreeBase有一定差距，主要体现在他的非结构化，直接从文本中直接提取出来，有较少的人工干预。所以在实体数目相同的情况下，有更多的relationship。

### Answering Reading Comprehension Using Memory Networks
主要学习其针对不同任务是如何微调模型以解决问题的方法。
1. bAbI数据集，这个之前介绍过，包含20个小任务，有些只需要简单推理，有些比较复杂，可能会涉及时序等多轮推理
2. MCTest，这是微软推出的阅读理解数据集，比较像我们高中的时候做的阅读理解，一段材料一个问题，四个选项。相比bAbI而言，是一种开放域数据集，涉及面比较广，自然语言表示，而且数据量更大（从文章长度、包含词数目等方面），而且是有选项的，这点需要注意。
3. WiKi QA，相比前面两个阅读理解型的数据而言，这个更倾向于QA。它是由很多Wikipedia文章组成的背景知识（40000+的vocab），而且每篇文章都比较长。在此之上提出了很多QA对。跟上面那篇论文所提出的Simple QA不一样的地方在于，这里的QA对不是基于知识库，而是基于Wikipedia 文章，相比结构化的知识库而言，显然直接从文章之中发掘答案更加困难。跟上面两个数据集的难度更不在一个数量级上面。

![](https://pic4.zhimg.com/80/v2-b2da1256a7a2dc283f0b1f9a747d84ff_hd.jpg)

### QA常用数据集
![](https://pic1.zhimg.com/80/v2-c8d5acb039906c19a92b3a30edb6b910_hd.jpg)

## [Key-Value Memory Networks](https://zhuanlan.zhihu.com/p/29817459)
本文在end-to-end的基础上对模型结构进行修改，使其可以更好的存储QA所需要的先验知识（KB或者wiki文章等


