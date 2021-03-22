# 谷歌、阿里、微软等10大深度学习CTR模型最全演化图谱【推荐、广告、搜索领域】
[参考网址](https://zhuanlan.zhihu.com/p/63186101)
本文总结了广告、推荐领域最为流行的10个深度学习CTR模型的结构特点，构建了它们之间的演化图谱。选择模型的标准尽量遵循下面三个原则：

1. 模型的在业界影响力较大的；
2. 已经被Google，微软，阿里等知名互联网公司成功应用的；
3. 工程导向的，而不是仅用实验数据验证或学术创新用的。

下面首先列出这张深度学习CTR模型的演化图谱，再对其进行逐一介绍：
![](https://pic2.zhimg.com/80/v2-763b523bd17349cd6cfecae2765db3d5_hd.jpg)

## 微软Deep Crossing（2016年）——深度学习CTR模型的base model
![](https://pic2.zhimg.com/80/v2-cef2b96858c05b98d698786884cfe891_hd.jpg)

微软于2016年提出的Deep Crossing可以说是深度学习CTR模型的最典型和基础性的模型。如图2的模型结构图所示，它涵盖了深度CTR模型最典型的要素，**即通过加入embedding层将稀疏特征转化为低维稠密特征，用stacking layer，或者叫做concat layer将分段的特征向量连接起来，再通过多层神经网络完成特征的组合、转换，最终用scoring layer完成CTR的计算。**跟经典DNN有所不同的是，Deep crossing采用的multilayer perceptron是由**残差网络**组成的，这无疑得益于MSRA著名研究员何恺明提出的著名的152层ResNet。

## 二、FNN（2016年）——用FM的隐向量完成Embedding初始化
![](https://pic2.zhimg.com/80/v2-658066ad055f48a6f03b7d2bc554368d_hd.jpg)
FNN相比Deep Crossing的创新在于使用FM的隐层向量作为user和item的Embedding，从而避免了完全从随机状态训练Embedding。由于id类特征大量采用one-hot的编码方式，导致其维度极大，向量极稀疏，所以Embedding层与输入层的连接极多，梯度下降的效率很低，这大大增加了模型的训练时间和Embedding的不稳定性，使用pre train的方法完成Embedding层的训练，无疑是降低深度学习模型复杂度和训练不稳定性的有效工程经验。

## 三、PNN (2016年)——丰富特征交叉的方式
![](https://pic4.zhimg.com/80/v2-ab2009fd2a0fbbac85f71aedd5cd34cb_hd.jpg)

PNN的全称是Product-based Neural Network，PNN的关键在于在embedding层和全连接层之间加入了Product layer。传统的DNN是直接通过多层全连接层完成特征的交叉和组合的，但这样的方式缺乏一定的“针对性”。首先全连接层并没有针对不同特征域之间进行交叉；其次，全连接层的操作也并不是直接针对特征交叉设计的。但在实际问题中，特征交叉的重要性不言而喻，比如年龄与性别的交叉是非常重要的分组特征，包含了大量高价值的信息，我们急需深度学习网络能够有针对性的结构能够表征这些信息。因此PNN通过加入Product layer完成了针对性的特征交叉，其product操作在不同特征域之间进行特征组合。并定义了inner product，outer product等多种product的操作捕捉不同的交叉信息，增强模型表征不同数据模式的能力 。

## 四、Google Wide&Deep（2016年）——记忆能力和泛化能力的综合权衡
![](https://pic2.zhimg.com/80/v2-894fb56966e758edf0eacf24f2869199_hd.jpg)

Google Wide&Deep模型的主要思路正如其名，把单输入层的Wide部分和经过多层感知机的Deep部分连接起来，一起输入最终的输出层。其中Wide部分的主要作用是让模型具有记忆性（Memorization），单层的Wide部分善于处理大量稀疏的id类特征，便于让模型直接“记住”用户的大量历史信息；Deep部分的主要作用是让模型具有“泛化性”（Generalization），利用DNN表达能力强的特点，挖掘藏在特征后面的数据模式。最终利用LR输出层将Wide部分和Deep部分组合起来，形成统一的模型。Wide&Deep对之后模型的影响在于——大量深度学习模型采用了两部分甚至多部分组合的形式，利用不同网络结构挖掘不同的信息后进行组合，充分利用和结合了不同网络结构的特点。

## 五、华为 DeepFM (2017年)——用FM代替Wide部分
![](https://pic1.zhimg.com/80/v2-226f6c7f0524df64c8f204869fe5e240_hd.jpg)

在Wide&Deep之后，诸多模型延续了双网络组合的结构，DeepFM就是其中之一。DeepFM对Wide&Deep的改进之处在于，它用FM替换掉了原来的Wide部分，加强了浅层网络部分特征组合的能力。事实上，由于FM本身就是由一阶部分和二阶部分组成的，DeepFM相当于同时组合了原Wide部分+二阶特征交叉部分+Deep部分三种结构，无疑进一步增强了模型的表达能力。

## 六、Google Deep&Cross（2017年）——使用Cross网络代替Wide部分
![](https://pic4.zhimg.com/80/v2-ddbe542944bc8bff8720c702537e6bbb_hd.jpg)

Google 2017年发表的Deep&Cross Network（DCN）同样是对Wide&Deep的进一步改进，主要的思路使用Cross网络替代了原来的Wide部分。其中设计Cross网络的基本动机是为了增加特征之间的交互力度，使用多层cross layer对输入向量进行特征交叉。单层cross layer的基本操作是将cross layer的输入向量xl与原始的输入向量x0进行交叉，并加入bias向量和原始xl输入向量。DCN本质上还是对Wide&Deep Wide部分表达能力不足的问题进行改进，与DeepFM的思路非常类似

## 七、NFM（2017年）——对Deep部分的改进
![](https://pic2.zhimg.com/80/v2-ce70760e88ca236e3d13f381df66cc4d_hd.jpg)

相对于DeepFM和DCN对于Wide&Deep Wide部分的改进，NFM可以看作是对Deep部分的改进。NFM的全称是Neural Factorization Machines，如果我们从深度学习网络架构的角度看待FM，FM也可以看作是由单层LR与二阶特征交叉组成的Wide&Deep的架构，与经典W&D的不同之处仅在于Deep部分变成了二阶隐向量相乘的形式。再进一步，NFM从修改FM二阶部分的角度出发，用一个带Bi-interaction Pooling层的DNN替换了FM的特征交叉部分，形成了独特的Wide&Deep架构。其中Bi-interaction Pooling可以看作是不同特征embedding的element-wise product的形式。这也是NFM相比Google Wide&Deep的创新之处。

## 八、AFM（2017年）——引入Attention机制的FM
![](https://pic4.zhimg.com/80/v2-07220b8851520e447a6336e897a0bf5b_hd.jpg)

AFM的全称是Attentional Factorization Machines，通过前面的介绍我们很清楚的知道，FM其实就是经典的Wide&Deep结构，其中Wide部分是FM的一阶部分，Deep部分是FM的二阶部分，而AFM顾名思义，就是引入Attention机制的FM，具体到模型结构上，AFM其实是对FM的二阶部分的每个交叉特征赋予了权重，这个权重控制了交叉特征对最后结果的影响，也就非常类似于NLP领域的注意力机制（Attention Mechanism）。为了训练Attention权重，AFM加入了Attention Net，利用Attention Net训练好Attention权重后，再反向作用于FM二阶交叉特征之上，使FM获得根据样本特点调整特征权重的能力。

## 九、阿里DIN（2018年）——阿里加入Attention机制的深度学习网络
![](https://pic3.zhimg.com/80/v2-8bbb5774eff2e079832536c45ed0f012_hd.jpg)

AFM在FM中加入了Attention机制，2018年，阿里巴巴正式提出了融合了Attention机制的深度学习模型——Deep Interest Network。与AFM将Attention与FM结合不同的是，DIN将Attention机制作用于深度神经网络，在模型的embedding layer和concatenate layer之间加入了attention unit，使模型能够根据候选商品的不同，调整不同特征的权重。

## 十、阿里DIEN（2018年）——DIN的“进化”
![](https://pic4.zhimg.com/80/v2-ba9a7cd89482001b79c37b845615db07_hd.jpg)

DIEN的全称为Deep Interest Evolution Network，它不仅是对DIN的进一步“进化”，更重要的是DIEN通过引入序列模型 AUGRU模拟了用户兴趣进化的过程。具体来讲模型的主要特点是在Embedding layer和Concatenate layer之间加入了生成兴趣的Interest Extractor Layer和模拟兴趣演化的Interest Evolving layer。其中Interest Extractor Layer使用了DIN的结构抽取了每一个时间片内用户的兴趣，Interest Evolving layer则利用序列模型AUGRU的结构将不同时间的用户兴趣串联起来，形成兴趣进化的链条。最终再把当前时刻的“兴趣向量”输入上层的多层全连接网络，与其他特征一起进行最终的CTR预估。

## 总结
1. 没有银弹。从来没有一个深度学习模型能够在所有数据集上都表现最优，特别是推荐、广告领域，各家的数据集，数据pattern、业务领域差异巨大，不存在能够解决一切问题的“银弹”模型。比如，阿里的DIEN对于数据质量、用户整个life cycle行为完整性的要求很高，如果在某些DSP场景下运用这个模型，预计不会收到良好的效果。再比如Google 的Deep&Cross，我们也要考虑自己的数据集需不需要如此复杂的特征交叉方式，在一些百万量级的数据集上，也许浅层神经网络的表现更好。
2. 算法工程师永远要在理想和现实间做trade off。有一种思想要避免，就是我为了要上某个模型就要强转团队的技术栈，强买某些硬件设备。模型的更新过程一定是迭代的，一定是从简单到复杂的，一定是你证明了某个模型是work的，然后在此基础上做改进的。这也是我们要熟悉所有模型演化关系的原因。