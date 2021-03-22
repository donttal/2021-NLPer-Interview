# NLP的bert和后bert时代的挑战
[toc]

## 一、迁移学习与模型预训练
迁移学习分类
把我们当前要处理的NLP任务叫做T（T称为目标任务），迁移学习技术做的事是利用另一个任务S（S称为源任务）来提升任务T的效果，也即把S的信息迁移到T中。至于怎么迁移信息就有很多方法了，可以直接利用S的数据，也可以利用在S上训练好的模型，等等。
依据目标任务T是否有标注数据，可以把迁移学习技术分为两大类，每个大类里又可以分为多个小类。
第一大类是T没有任何标注数据，比如现在很火的无监督翻译技术。但这类技术目前主要还是偏学术研究，离工业应用还有挺长距离的。工业应用中的绝大部分任务，我们总是能想办法标注一些数据的。而且，目前有监督模型效果要显著优于无监督模型。所以，面对完全没有标注数据的任务，最明智的做法是先借助于无监督技术（如聚类/降维）分析数据，然后做一些数据标注，把原始的无监督任务转变为有监督任务进行求解。基于这些原因，本文不再介绍这大类相关的工作。

第二大类是T有标注数据，或者说T是个有监督任务。这类迁移学习技术又可以依据源任务是否有监督，以及训练顺序两个维度，大致分为四小类：

源任务S是无监督的，且源数据和目标数据同时用于训练：此时主要就是自监督（self-supervised）学习技术，代表工作有之后会讲到的CVT。
源任务S是有监督的，且源数据和目标数据同时用于训练：此时主要就是多任务（multi-task）学习技术，代表工作有之后会讲到的MT-DNN。
源任务S是无监督的，且先使用源数据训练，再使用目标数据训练（序贯训练）：此时主要就是以BERT为代表的无监督模型预训练技术，代表工作有ELMo、ULMFiT、GPT/GPT-2、BERT、MASS、UNILM。
源任务S是有监督的，且先使用源数据训练，再使用目标数据训练（序贯训练）：此时主要就是有监督模型预训练技术，类似CV中在ImageNet上有监督训练模型，然后把此模型迁移到其他任务上去的范式。代表工作有之后会讲到的CoVe。
![](https://pic1.zhimg.com/80/v2-c558f4340c1be82a9f6007f47fb5c318_hd.jpg)

## 现状分析
先说说上表中四个类别的各自命运。以BERT为代表的无监督模型预训练技术显然是最有前途的。之前也说了，NLP中最不缺的就是无监督数据。只要堆计算资源就能提升效果的话，再简单不过了。

而无监督预训练的成功，也就基本挤压掉了自监督学习提升段位的空间。这里说的自监督学习不是泛指，而是特指同时利用无监督数据和当前有监督数据一起训练模型的方式。既然是同时训练，就不太可能大规模地利用无监督数据（要不然就要为每个特定任务都训练很久，不现实），这样带来的效果就没法跟无监督预训练方式相比。但自监督学习还是有存在空间的，**比如现在发现在做有监督任务训练时，把语言模型作为辅助损失函数加入到目标函数中，可以减轻精调或多任务学习时的灾难性遗忘（Catastrophic Forgetting）问题，提升训练的收敛速度。**所以有可能在训练时加入一些同领域的无监督数据，不仅能减轻遗忘问题，还可能因为让模型保留下更多的领域信息而提升最终模型的泛化性。但这个方向迎来大的发展可能性不大。

而类似CV中使用大规模有监督数据做模型预训练这条路，看着也比较暗淡，它自己单独不太可能有很大前景。**几个原因：1) 这条路已经尝试了很久，没有很显著的效果提升。2) NLP中获取大规模标注数据很难，而且还要求对应任务足够复杂以便学习出的模型包含各种语言知识。虽然机器翻译任务很有希望成为这种任务，但它也存在很多问题，比如小语种的翻译标注数据很少，翻译标注数据主要还是单句形式，从中没法学习到背景信息或多轮等信息。**但从另一个方面看，NLP搞了这么久，其实还是积累了很多标注或者结构化数据，比如知识图谱。如何把这些信息融合到具体任务中最近一直都是很活跃的研究方向，相信将来也会是。只是BERT出来后，这种做法的价值更像是打补丁，而不是搭地基了。

多任务学习作为代价较小的方法，前景还是很光明的。多个同领域甚至同数据上的不同任务同时训练，不仅能降低整体的训练时间，还能降低整体的预测时间（如果同时被使用），还能互相提升效果，何乐而不为。当然，多任务学习的目标一开始就不是搭地基。

上面说了这么多，其实想说的重点在下面。这些技术不一定非要单独使用啊，组合起来一起用，取长补短不是就皆大欢喜了嘛。
先回顾下现在的无监督模型预训练流程，如下图：

![](https://pic3.zhimg.com/80/v2-be2b1a37d4f3116a9bd039f496bbf7c6_hd.jpg)

首先是利用大的无监督数据预训练通用模型，优化目标主要是语言模型（或其变种）。第二步，利用有监督数据精调上一步得到的通用模型。这么做的目的是期望精调以后的通用模型更强调这个特定任务所包含的语言信息。这一步是可选的（所以图中对应加了`括号`），有些模型框架下没有这个步骤，比如BERT里面就没有。第三步才是利用有监督数据中对应的标注数据训练特定任务对应的模型。
那这个流程接下来会怎么发展呢？

## 未来可期
上面我已经对四类方法做了分别的介绍，包括对它们各自前途的简单判断，也介绍了当下效果最好的模型预训练流程。相信未来NLP的很多工作都会围绕这个流程的优化展开。我判断这个流程会继续发展为下面这个样子：
![](https://pic1.zhimg.com/80/v2-9d8f4fa719b92a661ab3ca262a9780a0_hd.jpg)

详细说明下每个步骤：
1. 第一步还是利用大的无监督数据预训练通用模型。但这里面目前可以改进的点有很多，比如发展比Transformer更有效的特征抽取结构，现在的Evolved Transformer和Universal Transformer等都是这方面的探索。发展更有效更多样化的预训练模型目标函数。目前预训练模型的目标函数主要是(Masked) LM和Next Sentence Prediction (NSP)，还是挺单一的。面向文档级背景或多轮这种长文本信息，未来应该会发展出更好的目标函数。比如有可能会发展出针对多轮对话这种数据的目标函数。

   BERT主要面向的是NLU类型的任务，目前微软提出的MASS、UNILM从不同的角度把BERT框架推广到NLG类型的任务上了，细节我们之后会讲到。GPT-2利用更大的模型获得了更好的语言模型。更多更好的数据，更大的模型带来的改进有没有极限？目前还不知道，相信很多公司已经在做这方面的探索了。但这个游戏目前还是只有大公司能玩得起，训练通用大模型太耗钱了。提升训练效率，很自然的就是另一个很重要的优化方向。
   
1. 第二步是利用其他大任务的标注数据或已有结构化知识精调第一步获得的通用模型。这一步不一定以单独的形式存在，它也可以放到第一步中，在预训练通用模型时就把这些额外信息注入进去，比如百度的ERNIE就是在预训练时就把实体信息注入进去了。既然人类在漫长的AI研究史上积累了大量各式各样的结构化数据，比如机器翻译标注数据，没理由不把它们用起来。相信未来会有很多知识融合（注入）这方面的工作。

2. 第三步和前面流程的第二步相同，即利用当前任务数据进一步精调上一步得到的通用模型。这么做的目的是期望精调后的模型更强调这个特定任务所包含的语言信息。ELMo的实验结论是，加入这一步往往能提升下一步的特定任务有监督训练的收敛速度，但仅在部分任务上最终模型获得了效果提升（在另一部分任务上持平）。

   另一种做法是把这一步与下一步的特定任务有监督训练放在一块进行，也即在特定任务有监督训练时把**语言模型作为辅助目标函数加入到训练过程中**，以期提升模型收敛速度，降低模型对已学到知识的遗忘速度，提升最终模型的效果。GPT的实验结论是，如果特定任务有监督训练的数据量比较大时，加入辅助语言模型能改善模型效果，但如果特定任务有监督训练的数据量比较小时，加入辅助语言模型反而会降低模型效果。但ULMFiT上的结论刚好相反。。所以就试吧。
   
4. 利用多任务或者单任务建模方式在有监督数据集上训练特定任务模型。多任务的很多研究相信都能移植到这个流程当中。我们之后会介绍的微软工作MT-DNN就是利用BERT来做多任务学习的底层共享模型。论文中的实验表明加入多任务学习机制后效果有显著提升。相信在这个方向还会有更多的探索工作出现。在单任务场景下，原来大家发展出的各种任务相关的模型，是否能在无监督预训练时代带来额外的收益，这也有待验证。

## 二、各类代表性工作
套用下前面对迁移学习分类的方式，把接下来要介绍的具体模型放到对应的模块里，这样逻辑会更清楚一些。

![img](https://pic3.zhimg.com/80/v2-3f4489c43e9cc2e5e05ec683e6a456ae_hd.jpg)

我们先介绍CoVe和CVT。

### 有监督模型预训练：CoVe
CoVe是在 McCann et al., Learned in Translation: Contextualized Word Vectors 这个论文中提出的。自然语言中的一词多义非常常见，比如“苹果手机”和“苹果香蕉”里的“苹果”，含义明显不同。以Word2Vec为代表的词表示方法没法依据词所在的当前背景调整表示向量。所以NLPer一直在尝试找背景相关的词表示法（Contextualized Word Representation）。CoVe就是这方面的一个尝试。
CoVe首先在翻译标注数据上预训练encoder2decoder模型。其中的encoder模块使用的是BiLSTM。训练好的encoder，就可以作为特征抽取器，获得任意句子中每个token的带背景词向量：
$$\operatorname{CoVe}(x)=\operatorname{biLSTM}(\text { Glo } V e(x))$$
使用的时候，只要把$\operatorname{CoVe}(x)$和$\text { GloVe(x) }$拼接起来就行。

![](https://pic1.zhimg.com/80/v2-52ff5d14f67f88644244fe2ea87773c0_hd.jpg)

论文作者在分类和匹配下游任务对CoVe的效果做过验证，效果肯定是有一些提升了，但提升也不是很明显。

![img](https://pic4.zhimg.com/80/v2-f61bbe650671f4b893ab46fcff951073_hd.jpg)

总结下CoVe的特点：

- 预训练依赖于有监督数据（翻译数据）。
- CoVe结果以特征抽取的方式融合到下游任务模型中，但下游任务还是要自定义对应的模型。

### 自监督学习同时训练：CVT
CVT (Cross-View Training)在利用有监督数据训练特定任务模型时，同时会使用无监督数据做自监督学习。Encoder使用的是2层的CNN-BiLSTM，训练过程使用标注数据和非标注数据交替训练。利用标注数据训练主预测模块，同时构造多个辅助模块，**辅助模块利用非标注数据拟合主模块的预测概率。辅助模块的输入仅包含所有输入中的部分信息，**这个思想和dropout有点像，可以提高模型的稳定性。不同的特定任务，辅助模块的构造方式不同，如何选输入中部分信息的方式也不同。

![](https://pic4.zhimg.com/80/v2-24bc6b518171493b1af3819884e46a93_hd.jpg)

例如，对于序列标注任务，论文中以biLSTM第一层和第二层的状态向量拼接后输入进主预测模块。而4个辅助模块则使用了第一层的各个单向状态向量作为输入.$p_{\theta}^{\mathrm{fwd}}\left(y^{t} | x_{i}\right)$使用的是第一层前向LSTM当前词的状态向量， $p_{\theta}^{\mathrm{bwd}}\left(y^{t} | x_{i}\right)$使用的是第一层后向LSTM当前词的状态向量。 $p_{\theta}^{\mathrm{bwd}}\left(y^{t} | x_{i}\right)$使用的是第一层前向LSTM前一个词的状态向量，而$p_{\theta}^{\mathrm{past}}\left(y^{t} | x_{i}\right)$使用的是第一层后向LSTM后一个词的状态向量。

![](https://pic4.zhimg.com/80/v2-ec96e06690592a266fa5d1d8302a1e23_hd.jpg)

作者也在多任务学习上验证了CVT带来效果提升。CVT使用多个标注数据和非标注数据交替训练。使用标注数据训练时，CVT随机选择一个任务，优化对应任务的主模块目标函数。使用非标注数据训练时，CVT为所有任务产生对应的辅助模块。这些辅助模块同时被训练，相当于构造了一些所有任务共用的标注数据。这种共用的训练数据能提升模型收敛速度。作者认为效果提升的主要原因是，同时训练多个任务能降低模型训练一个任务时遗忘其他任务知识的风险。
总结下CVT的特点：

- 在训练特定任务模型时加入无监督数据做自监督学习，获得了精度的提升。其中辅助模块的构建是关键。
- 需要为不同任务定制不同的辅助模块。
- 应用于MTL问题效果比ELMo好。

### 无监督模型预训练ELMo
ELMo (Embedding from Language Models) 的目的是找到一种带背景的词向量表示方法，以期在不同的背景下每个词能够获得更准确的表示向量。
ELMo的使用过程分为以下三个步骤：
![img](https://pic4.zhimg.com/80/v2-fecc983cd6947f52372d5d0152d8297b_hd.jpg)

第一步是预训练阶段，ELMo利用2层的biLSTM和无监督数据训练两个单向的语言模型，它们统称为biLM。

![img](https://pic3.zhimg.com/80/v2-ea3b87319e3f2ff46721fa6f84f576c6_hd.jpg)

第二步利用特定任务的数据精调第一步得到的biLM。作者发现这步能显著降低biLM在特定任务数据上的PPL，结果如下图。但对特定任务最终的任务精度未必有帮助（但也不会降低任务精度）。作者发现在SNLI（推断）任务上效果有提升，但在SST-5（情感分析）任务上效果没变化。

![img](https://pic1.zhimg.com/80/v2-6350dcf54983b5120c6ae8bec96fb898_hd.jpg)

第三步是训练特定的任务模型。任务模型的输入是上面已训练biLM的各层状态向量的组合向量。
$$\mathbf{E L M o}_{k}^{\mathrm{task}}=E\left(R_{k} ; \Theta^{\mathrm{task}}\right)=\gamma^{\mathrm{task}} \sum_{j=0}^{L} s_{j}^{\mathrm{task}} \mathbf{h}_{k, j}^{L M}$$
其中$s_{j}^{\mathrm{task}}$是经过softmax归一化后的权重，$\gamma^{\text {task }}$是整体的scale参数。它们都是任务模型中待学习的参数。
$\mathbf{E} \mathbf{L} \mathbf{M} \mathbf{o}_{k}^{\mathrm{task}}$可以以额外特征的方式，加入到特定任务的输入和输出特征中。作者发现，对于某些任务，把$\mathbf{E} \mathbf{L} \mathbf{M} \mathbf{o}_{k}^{\mathrm{task}}$同时加入到输入和输出特征中效果最好，具体见下图。

![img](https://pic1.zhimg.com/80/v2-822cfa76fe538f47b102180e642859e8_hd.jpg)

作者发现，biLM底层LSTM的输出状态对句法任务（如POS）更有帮助，而高层LSTM的输出状态对语义任务（如WSD）更有帮助。ELMo对（标注）数据量少的有监督任务精度提升较大，对数据量多的任务效果提升就不明显了。这说明ELMo里存储的信息比较少，还是它主要功能是帮助有监督数据更好地提炼出其中的信息？
总结下ELMo的特点：

- 把无监督预训练技术成功应用于多类任务，获得了效果提升。
- ELMo以特征抽取的形式融入到下游任务中，所以不同下游任务依旧需要使用不同的对应模型。
- ELMo改进的效果依赖于下游任务和对应的模型，且改进效果也不是特别大。

### ULMFiT & SiATL
ULMFiT (Universal Language Model Fine-tuning) 使用和ELMo类似的流程：

1. 使用通用数据预训练LM，模型使用了3层的AWD-LSTM。
2. 在特定任务数据上精调LM，其中使用到**差异精调**和**倾斜三角lr**两个策略。
3. 以LM作为初始值，精调特定任务分类模型，其中使用到**逐层解冻**、**差异精调**和**倾斜三角lr**三个策略。经过AWD-LSTM之后，输出给分类器的向量为三个向量的拼接： 
$$\mathbf{h}_{c}=\left[\mathbf{h}_{T}, \operatorname{maxpool}(\mathbf{H}), \text { meanpool }(\mathbf{H})\right]$$

- 最后一层最后一个词对应的向量；
- 最后一层每个词向量做max pooling；
- 最后一层每个词向量做mean pooling。

![差异精调](https://pic1.zhimg.com/80/v2-4dfcceae42212c8aece6a4c7fc2f21e8_hd.jpg)

论文中提出了几个优化策略，能够提升精调后模型的最终效果。

![三角学习率](https://pic2.zhimg.com/80/v2-cf4e43cfac65ed75f635d730349ead11_hd.jpg)

![](https://pic2.zhimg.com/80/v2-b0b4749213d967b21242b4e9d5de0ec9_hd.jpg)

论文中的实验主要针对各种分类任务，相比于之前最好的结果，ULMFiT把分类错误率降低了18-24%。

![](https://pic3.zhimg.com/80/v2-abe9ff341c3acfb77a24312ded05047a_hd.jpg)

论文中也设计了实验来说明流程中第二步（在特定任务数据上精调LM）的作用。结果表明第二步的加入，能够让第三步的分类任务在很少的数据量下获得好的结果。只要使用 `1%~10%`的标注数据，就能达到不加第二步时的模型效果。

![img](https://pic3.zhimg.com/80/v2-4dc0f647f212bc8e49bdc52328e5aaa2_hd.jpg)

作者也设计了去除实验验证论文中提出的三个策略的效果：差异精调（discr）、倾斜三角lr（stlr）、逐层解冻（Freez）。结果表明相比于其他人提出的策略，这几个策略能获得更好的结果。而且，相比于不使用discr和stlr机制的精调策略（Full），ULMFiT模型更稳定，没出现灾难性遗忘。

![img](https://pic4.zhimg.com/80/v2-b5d4951b28b88219c42674517fcd7423_hd.jpg)

之后的另一篇论文 [An Embarrassingly Simple Approach for Transfer Learning from Pretrained Language Models](https://arxiv.org/abs/1902.10547) 建议了一些新的策略，解决精调时的灾难性遗忘问题。模型称为**SiATL** (Single-step Auxiliary loss Transfer Learning)。SiATL只包含两个步骤：无监督数据预训练LM、精调分类模型。但在精调分类模型时，SiATL把LM作为辅助目标加入到优化目标函数当中。SiATL的第二步相当于把ULMFiT的第二步和第三步一起做了。所以它们的流程其实是一样的。

![img](https://pic4.zhimg.com/80/v2-e866fd3031e99132da585f199b7b1deb_hd.jpg)

预训练模型使用的是两层LSTM+Linear，而分类模型在预训练模型的上面增加了一层带self-attention的LSTM和输出层。SiATL建议的几个策略：

![img](https://pic3.zhimg.com/80/v2-33f8d0c988653a313779af0ae6418246_hd.jpg)

论文发现辅助LM目标对于小数据集更有用，可能是辅助LM减轻了小数据集上的过拟合问题。其中的系数 $\gamma$ ，论文实验发现初始取值为 `0.2`，然后指数下降到 `0.1`效果最好。 $\gamma$ 的取值需要考虑到 $L_{t a s k}$ 和 $L_{L M}$ 的取值范围。这个结论和ULMFiT中验证第二步流程作用的实验结果相同，也侧面说明了它们本质上差不多。
**另一个发现是如果预训练用的无监督数据和任务数据所在领域不同，序贯解冻带来的效果更明显。这也是合理的，领域不同说明灾难性遗忘问题会更严重，所以迁移知识时要更加慎重，迁移过程要更慢。序贯解冻主要就是用途就是减轻灾难性遗忘问题。**
论文还发现，和ULMFiT相比，SiATL在大数据集上效果差不多，但在小数据集要好很多。

![img](https://pic3.zhimg.com/80/v2-f7486a035815db8ce224dc757e17fa9a_hd.jpg)

总结下 ULMFiT 和 SiATL：

- ULMFiT使用序贯训练的方式组合特定任务LM和任务目标函数，而SiATL使用同时训练的方式，也即加入辅助LM目标函数。
- **它们建议的策略都是在解决灾难性遗忘问题，也都解决的不错。可以考虑组合使用这些策略。**
- 它们在小数据集上都提升明显，只要使用 1%~10% 的标注数据，就能达到之前的效果。
- 虽然它们只在分类任务上验证了各自的效果，但这些策略应该可以推广到其他任务上。

### GPT/GPT-2
前面介绍的工作中预训练模型用的都是多层LSTM，而OpenAI GPT首次使用了Transformer作为LM预训练模型。**GPT使用12层的Transformer Decoder训练单向LM，也即mask掉当前和后面的词。**
在做精调时，使用最高层最后一个词的向量作为后续任务的输入，类似SiATL也加入了辅助LM目标函数。

![img](https://pic3.zhimg.com/80/v2-8c8f47a514284f71ba19d287378a98fe_hd.jpg)

GPT的另一个大贡献是为下游任务引入了统一的模型框架，也即不再需要为特定任务定制复杂的模型结构了。不同的任务只需把输入数据做简单的转换即可。

![img](https://pic2.zhimg.com/80/v2-d0f76fb9834219aa54b69656733c6481_hd.jpg)

GPT在多种类型的任务上做了实验，12个任务中的9个任务有提升，最高提升幅度在9%左右，效果相当不错。
针对预训练、辅助LM和Transformer，论文中做了去除实验，结果表明预训练最重要，去掉会导致指标下降14.8%，而Transformer改为LSTM也会导致指标下降5.6%。比较诡异的是去掉辅助LM的实验结果。去掉辅助LM，只在QQP (Quora Question Pairs)和NLI上导致指标下降。在其他任务上反而提升了指标。作者观察到的趋势是辅助LM对于大的数据集比小的数据集更有帮助。。这也跟ULMFiT和SiATL中的结论相反。

![img](https://pic3.zhimg.com/80/v2-11cda2555d74ad51371015ba6518e24a_hd.jpg)

**总结下GPT的主要贡献：

- 验证了Transformer在Unsupervised Pretraining中的有效性。
- 验证了更大的模型效果更好： 6 --> 12 层。
- 为下游任务引入了通用的求解框架，不再为任务做模型定制。**

之后OpenAI又训练一个更大的模型，叫GPT-2。GPT-2把GPT中12层的Transformer提升到48层，参数数量是GPT的十几倍，达到了15亿。
GPT-2依旧使用单向LM训练语言模型，但使用数量更多、质量更好、覆盖面更广的数据进行训练。而且，GPT-2没有针对特定模型的精调流程了。作者想强调的是，预训练模型中已经包含很多特定任务所需的信息了，只要想办法把它们取出来直接用即可，可以不用为特定任务标注数据，真正达到通用模型的能力。
那，没有精调如何做特定任务呢？一些任务说明如下：

![img](https://pic1.zhimg.com/80/v2-a9c24abbf68b88cfd6df7d9eae77b160_hd.jpg)

不做精调的GPT-2不仅在很多特定任务上已经达到了SOTA，还在生成任务上达到了吓人的精度。

![img](https://pic1.zhimg.com/80/v2-e6991722eb63407389cb397511a0fb7c_hd.jpg)

![img](https://pic1.zhimg.com/80/v2-0d71bc24f0370939c9861dd326a2bdd0_hd.jpg)

### BERT
和GPT一样，BERT的基本模型使用了Transformer，只是模型又变大了（12层变成了24层）。

![img](https://pic4.zhimg.com/80/v2-2ccb0ab23c6f848cb2cba70d96c21477_hd.jpg)

**相比于GPT的单向LM，BERT使用了双向LM。但显然预测时不能让待预测的词看到自己，所以需要把待预测词mask掉。BERT建议了masked LM机制，即随机mask输入中的 `k%`个词，然后利用双向LM预测这些词。**

![img](https://pic2.zhimg.com/80/v2-c9d262f9dacb985b5825249d3a511549_hd.png)

但mask时需要把握好度。mask太少的话，训练时每次目标函数中包含的词太少，训练起来就要迭代很多步。mask太多的话，又会导致背景信息丢失很多，与预测时的情景不符。而且，简单的mask会带来预训练和精调训练的不一致性：精调阶段，输入数据里是不mask词的。
BERT建议了以下的策略，解决这些问题：

![img](https://pic1.zhimg.com/80/v2-62758300dbaf44da047d8a52c35ef884_hd.jpg)

**BERT的另一大贡献，是引入了新的预训练目标 **Next Sentence Prediction (NSP)** 。对于两个句子A和B，NSP预测B是不是A的下一个句子。训练时NSP的正样本就是从文档从随机选的两个临近句子，而负样本就是B是随机从文档中选取的，与A的位置没关系。NSP可以学习句子与句子间的关系。
预训练的目标函数是Masked LM和NSP的加和。**

![img](https://pic3.zhimg.com/80/v2-49d8d76f0a73063797d151c691f65272_hd.jpg)

**BERT的输入词向量是三个向量之和：

- Token Embedding：WordPiece tokenization subword词向量。
- Segment Embedding：表明这个词属于哪个句子（NSP需要两个句子）。
- Position Embedding：学习出来的embedding向量。这与Transformer不同，Transformer中是预先设定好的值。**

![img](https://pic1.zhimg.com/80/v2-a10253c8dab967e94f4dfc336a95c6f8_hd.jpg)

BERT也为下游任务引入了通用的求解框架，不再为任务做模型定制。对于分类和匹配任务，下游任务只要使用第一个词 `[CLS]`对应的最上层输出词向量作为分类器的输入向量即可。对于抽取式QA和序列标注问题，使用每个词对应的最上层输出词向量作为下游任务的输入即可。

![img](https://pic4.zhimg.com/80/v2-842fd7d7d68931df807de8ddf5b41db3_hd.jpg)

BERT的惊艳结果，引爆了NLP行业。BERT在11个任务上获得了最好效果，GLUE上达到了80.4%，提升了整整7.6个点，把SQuAD v1.1 F1又往上提升了1.5个点，达到了93.2 。
BERT的去除实验表明，双向LM和NSP带了的提升最大。

![img](https://pic4.zhimg.com/80/v2-4a97d6a6f6a9da7284d89a07fbf3e9f7_hd.jpg)

另一个结论是，增加模型参数数量可以提升模型效果。

![img](https://pic4.zhimg.com/80/v2-895f9b9d7e805a68b6156bbf043d450b_hd.jpg)

BERT预训练模型的输出结果，无非就是一个或多个向量。**下游任务可以通过精调（改变预训练模型参数）或者特征抽取（不改变预训练模型参数，只是把预训练模型的输出作为特征输入到下游任务）两种方式进行使用。**BERT原论文使用了精调方式，但也尝试了特征抽取方式的效果，比如在NER任务上，最好的特征抽取方式只比精调差一点点。但特征抽取方式的好处可以预先计算好所需的向量，存下来就可重复使用，极大提升下游任务模型训练的速度。

![img](https://pic2.zhimg.com/80/v2-724018a6eec1e185157333fc2b9812cd_hd.jpg)

后来也有其他人针对ELMo和BERT比较了这两种使用方式的精度差异。下面列出基本结论：

![img](https://pic3.zhimg.com/80/v2-5d287c93a7c99761c0416f8289fde0fe_hd.jpg)

![img](https://pic1.zhimg.com/80/v2-108b55ae9647cd87e2a44fd11594ff38_hd.jpg)

**总结下BERT的主要贡献：

- 引入了Masked LM，使用双向LM做模型预训练。
- 为预训练引入了新目标NSP，它可以学习句子与句子间的关系。
- 进一步验证了更大的模型效果更好： 12 --> 24 层。
- 为下游任务引入了很通用的求解框架，不再为任务做模型定制。
- 刷新了多项NLP任务的记录，引爆了NLP无监督预训练技术。**

### MASS
BERT只能做NLU类型的任务，无法直接用于文本产生式（NLG）类型的任务，如摘要、翻译、对话生成。NLG的基本框架是encoder2decoder，微软的MASS (MAsked Sequence to Sequence pre-training)把BERT推广到NLG任务。MASS的结构如下，它的训练数据依旧是单句话，但是会随机mask这句话中连续的 `k`个词，然后把这些词放入decoder模块的相同位置，而encoder中只保留未被mask的词。MASS期望decoder利用encoder的信息和decoder前面的词，预测这些被mask的词。

![img](https://pic4.zhimg.com/80/v2-f2a0421126335896a5fc22a599c04703_hd.jpg)

比较有意思的是，BERT和GPT都是MASS的特例。当 `k=1`时，也即随机mask单个词时，MASS就退化成BERT；当 `k=句子长度` 时，也即mask所有词时，MASS就退化成GPT，或者标准的单向LM。

![img](https://pic4.zhimg.com/80/v2-5243d3595b650bd418b57222d719f5fb_hd.png)

论文中使用了4层的Transformer作为encoder和decoder，**跟encoder使用BERT，decoder
使用标准单向LM的框架BERT+LM做了效果对比，PPL上降低了不少。而且作者也对比了 `k` 取不同值时的效果变化，结果发现在多个任务上它取50%句子长度都是最优的。**

![img](https://pic4.zhimg.com/80/v2-a124387acf9acc56be3ac19812ada6df_hd.jpg)

**为什么MASS能取得比较好的效果？作者给出了以下解释：

- Encoder中mask部分tokens，迫使它理解unmasked tokens。
- Decoder中需要预测masked的连续tokens，让decoder能获取更多的语言信息。
- Decoder中只保留了masked的tokens，而不是所有的tokens，迫使decoder也会尽量从encoder中抽取信息。**

作者也做了两个去除实验验证上面的后两条解释。

![img](https://pic4.zhimg.com/80/v2-59bc7c5c6d23f8d1907c076f3aad82a7_hd.jpg)

**总结下MASS的特点：

- 把BERT推广到NLG类型任务，并且统一了BERT和传统单向LM框架。
- 实验表明MASS效果比BERT+LM好，但实验使用的模型太小，不确定这种优势在模型变大后是否还会存在。**

### UNILM
UNILM (UNIfied pretrained Language Model)是微软另一波人最近放出的论文。UNILM同时训练BERT中的双向LM、GPT中的单向LM和seq2seq中的LM。用的方法也很自然，核心思想在Transformer那篇论文中其实就已经在用了。
UNILM中的核心框架还是Transformer，只是用无监督数据预训练模型时，同时以双向LM、单向LM和seq2seq LM为目标函数。这些目标函数共享一个Transformer结构，训练也都使用了类似BERT中的 `[MASK]`机制。

![img](https://pic4.zhimg.com/80/v2-4fd057426c89df8f64e7f573fa3d16f3_hd.jpg)

和BERT的双向LM不同的是，单向LM在做self-attention时不能使用这个词后面的词。seq2seq LM在做decoder 预测时也有类似的约束，做self-attention时能使用encoder中的所有词，以及decoder中当前词（替换为 `[MASK]`了）和前面的词，而不能使用decoder中这个词后面的词。UNILM在做self-attention时通过mask机制来满足这些约束，也即在softmax函数中把后面词对应的向量元素值改为 `-∞`。
seq2seq LM是把两个句子拼起来（和BERT相同）直接输入一个Transformer（只是预测encoder和decoder中被mask的词时，对self-attention使用了不同的约束条件），所以encoder和decoder使用的是同一个Transformer。seq2seq LM的训练样本，和NSP任务类似，正样本为连续的两个句子，负样本为随机的两个句子。
对词随机mask的机制和BERT类似，只是会以一定概率mask临近的两个或三个词，具体说明如下：

![img](https://pic2.zhimg.com/80/v2-ff5f00c0c77a3c305ece17ffd1537eb5_hd.jpg)

训练时目标函数的设定也参照BERT，只是要同时兼顾双向LM、单向LM和seq2seq LM。作者使用的模型大小同 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BBERT%7D_%7Blarge%7D) ，也即用了24层的Transformer。

![img](https://pic3.zhimg.com/80/v2-39f1e97371dc4466df8841edd79ba9ea_hd.jpg)

精调阶段，对于NLU类型的任务UNILM和BERT相同。对于NLG类型的任务，UNILM随机mask decoder中的一些词，然后再预测它们。以下是UNILM应用于生成式QA任务的做法，效果提升很明显。

![img](https://pic4.zhimg.com/80/v2-2555646d1b3655602937bed5fa192cff_hd.jpg)

**对于GLUE的所有任务，UNILM据说是首次不添加外部数据打赢BERT的模型！
**
![img](https://pic1.zhimg.com/80/v2-1f41772aeef5600a0cc6bdbe13a3d51c_hd.jpg)

**总结下UNILM的特点：

- 预训练同时训练双向LM、单向LM和seq2seq LM，使用 `mask`机制解决self-attention中的约束问题。
- 可以处理NLU和NLG类型的各种任务。
- 在GLUE上首次不加外部数据打赢了BERT。**


## 多任务学习：MT-DNN
MT-DNN (Multi-Task Deep Neural Network)是去年年底微软的一篇工作，思路很简单，就是在MTL中把BERT引入进来作为底层共享的特征抽取模块。

![img](https://pic1.zhimg.com/80/v2-0cd5146ae2728e1d78703e00f27bbc9c_hd.jpg)

预训练就是BERT，精调时每个batch随机选一个任务进行优化。整体算法步骤如下：

![img](https://pic3.zhimg.com/80/v2-19814f650a1c105fadbb6f38f0b6ce7e_hd.jpg)

**MT-DNN在GLUE上效果比BERT好不少，当然主要原因可能是加入了额外的数据了。作者也对比了多任务与单任务的结果，多任务确实能给每个任务都带来效果提升。
**
![img](https://pic2.zhimg.com/80/v2-419a789b6a81bdef737b03cb9e915c45_hd.jpg)

总结下MT-DNN的特点：

- 框架简单明了：MT-DNN = BERT + MTL。


# 三、实践、观点、总结
实践与建议
虽然前面介绍的很多模型都能找到实现代码。**但从可用性来说，对于NLU类型的问题，基本只需考虑ELMo，ULMFiT和BERT。**而前两个没有中文的预训练模型，需要自己找数据做预训练。BERT有官方发布的中文预训练模型，很多深度学习框架也都有BERT的对应实现，而且BERT的效果一般是最好的。但BERT的问题是速度有点慢，使用12层的模型，对单个句子（30个字以内）的预测大概需要100~200毫秒。如果这个性能对你的应用没问题的话，建议直接用BERT。
对于分类问题，如果特定任务的标注数据量在几千到一两万，可以直接精调BERT，就算在CPU上跑几十个epoches也就一两天能完事，GPU上要快10倍以上。如果标注数据量过大或者觉得训练时间太长，可以使用特征抽取方式。先用BERT抽取出句子向量表达，后续的分类器只要读入这些向量即可。
我们目前在很多分类问题上测试了BERT的效果，确实比之前的模型都有提升，有些问题上提升很明显。下图给出了一些结果示例。

![img](https://pic1.zhimg.com/80/v2-39fabbc169fd4ad7c107ba3957c14508_hd.jpg)

BERT当然可以直接用来计算两个句子的匹配度，只要把query和每个候选句子拼起来，然后走一遍BERT就能算出匹配度。这样做的问题是，如果有100个候选结果，就要算100次，就算把它们打包一起算，CPU上的时间开销在线上场景也是扛不住的。但如果使用Siamese结构，我们就可以把候选句子的BERT向量表达预先算好，然后线上只需要计算query的BERT向量表达，然后再计算query和候选句子向量的匹配度即可，这样时间消耗就可以控制在200ms以内了。

![img](https://pic4.zhimg.com/80/v2-1df8ed37159ffe68ccf13993ea07a4e3_hd.jpg)

使用Siamese这种结构理论上会降低最终的匹配效果，之前也有相关工作验证过在一些问题上确实如此。我们目前在自己的三个数据上做了对比实验（见下图），发现在两个问题上效果确实略有下降，而在另一个问题上效果基本保持不变。我估计只要后续交互层设计的合理，Siamese结构不会比原始BERT精调差很多。

![img](https://pic2.zhimg.com/80/v2-98ba3bfb83f66a811ed02f213a9e07a1_hd.jpg)

# 观点
按理ELMo的想法很简单，也没什么模型创新，为什么之前就没人做出来然后引爆无监督模型预训练方向？BERT的一作Jacob Devlin认为主要原因是之前使用的数据不够多，模型不够大。无监督预训练要获得好效果，付出的代价需要比有监督训练大到1000到10w倍才能获得好的效果。之前没人想到要把数据和模型规模提高这么多。
为了让预训练的模型能对多种下游任务都有帮助，也即预训练模型要足够通用，模型就不能仅仅只学到带背景的词表示这个信息，还需要学到很多其他信息。而预测被mask的词，就可能要求模型学到很多信息，句法的，语义的等等。所以，相对于只解决某个下游特定任务，预训练模型要通用的话，就要大很多。目前发现只要使用更多（数量更多、质量更好、覆盖面更广）的无监督数据训练更大的模型，最终效果就会更优。目前还不知道这个趋势的极限在什么量级。
**BERT虽然对NLU的各类任务都提升很大，但目前依旧存在很多待验证的问题。比如如何更高效地进行预训练和线上预测使用，如何融合更长的背景和结构化知识，如何在多模态场景下使用，在BERT之后追加各种任务相关的模块是否能带来额外收益等等。**

最后，简单总结一下。
无监督预训练技术已经在NLP中得到了广泛验证。BERT成功应用于各种NLU类型的任务，但无法直接用于NLG类型的任务。微软最近的工作MASS把BERT推广到NLG类型任务，而UNILM既适用于NLU也适用于NLG任务，效果还比BERT好一点点。
相信未来NLP的很多工作都会围绕以下这个流程的优化展开：

![img](https://pic1.zhimg.com/80/v2-9d8f4fa719b92a661ab3ca262a9780a0_hd.jpg)

在这个过程中，我们还收获了诸多副产品：

- 相对于biLSTM，Transformers在知识抽取和存储上效果更好，潜力还可发掘。它们之间的具体比较，推荐俊林老师的“[放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)”，里面介绍的很清楚。

**- 目前无监督模型预训练常用以下几种目标函数：

- - **一般的LM**。基于token的交叉熵。
  - **Masked LM**。相比于一般的LM，masked LM能够使用双向tokens，且在模型训练和预测时的数据使用方式更接近，降低了它们之间的gap。
  - **Consecutive masked LM**。Mask时不仅随机mask部分离散的token，还随机mask一些连续的tokens，如bi-grams、tri-grams等。这种consecutive mask机制是否能带来普遍效果提升，还待验证。
  - **Next Sentence Prediction**。预测连续的两个句子是否有前后关系。

- 精调阶段，除了任务相关的目标函数，还可以考虑把LM作为辅助目标加到目标函数中。加入LM辅助目标能降低模型对已学到知识的遗忘速度，提升模型收敛速度，有些时候还能提升模型的精度。精调阶段，学习率建议使用linear warmup and linear decay机制，降低模型对已学到知识的遗忘速度。如果要精调效果，可以考虑ULMFiT中引入的gradual unfreezing和discriminative fine-tuning:机制。

- 使用数量更多、质量更好、覆盖面更广的无监督数据训练更大的模型，最终效果就会更优。目前还不知道这个趋势的极限在什么地方。**

**最后说一点自己的感想。**
NLP中有一部分工作是在做人类知识或人类常识的结构化表示。有了结构化表示后，使用时再想办法把这些表示注入到特定的使用场景中。比如知识图谱的目标就是用结构化的语义网络来表达人类的所有知识。这种结构化表示理论上真的靠谱吗？人类的知识真的能完全用结构化信息清晰表示出来吗？显然是不能，我想这点其实很多人都知道，只是在之前的技术水平下，也没有其他的方法能做的更好。所以这是个折中的临时方案。
无监督预训练技术的成功，说明语言的很多知识其实是可以以非结构化的方式被模型学习到并存储在模型中的，只是目前整个过程我们并不理解，还是黑盒。相信以后很多其他方面的知识也能找到类似的非结构化方案。所以我估计知识图谱这类折中方案会逐渐被替代掉。当然，这只是我个人的理解或者疑惑，仅供他人参考。

# 参考资料
[张俊林，【知乎专栏】深度学习前沿笔记](https://zhuanlan.zhihu.com/c_188941548)
[Lilian Weng,Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html)
[McCann et al.2017,Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107)
[ELMo: Deep Contextual Word Embeddings](https://arxiv.org/abs/1802.05365), AI2 & University of Washington, 2018
[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
Ashish Vaswani, et al. [“Attention is all you need.”](https://arxiv.org/abs/1706.03762) NIPS 2017
[Jacob Devlin, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://nlp.stanford.edu/seminar/details/jdevlin.pdf)
[自然语言处理中的语言模型预训练方法（ELMo、GPT和BERT）](https://hackernoon.com/to-tune-or-not-to-tune-adapting-pretrained-representations-to-diverse-tasks-paper-discussion-2dabe678ef83)
[To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks](https://hackernoon.com/to-tune-or-not-to-tune-adapting-pretrained-representations-to-diverse-tasks-paper-discussion-2dabe678ef83)
Kevin Clark et al. [“Semi-Supervised Sequence Modeling with Cross-View Training.”](https://arxiv.org/abs/1809.08370)EMNLP 2018
Matthew E. Peters, et al. [“Deep contextualized word representations.”](https://arxiv.org/abs/1802.05365) NAACL-HLT 2017
Jeremy Howard and Sebastian Ruder. [“Universal language model fine-tuning for text classification.”](https://arxiv.org/abs/1801.06146) ACL 2018
Alec Radford et al. [“Improving Language Understanding by Generative Pre-Training”](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). OpenAI Blog, June 11, 2018