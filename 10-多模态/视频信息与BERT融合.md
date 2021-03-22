# 视频图像等信息与BERT融合
[网址1](https://blog.csdn.net/abcdefg90876/article/details/104645488/)

[网址2](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/104628010)

其在多模态领域的应用主要分为了两个流派：一个是单流模型，在单流模型中文本信息和视觉信息在一开始便进行了融合；另一个是双流模型，在双流模型中文本信息和视觉信息一开始先经过两个独立的编码模块，然后再通过互相的注意力机制来实现不同模态信息的融合。

视频可以理解为一组快速播放的图片，其中每一幅图片定义为帧（frame）。一般处理视频数据首先需要按每秒钟x帧（fps）的频率去对视频做抽取，然后将n个连续的frame组成一个片段（clip），这样视频就被切割成了很多不重叠的片段。对于每一个片段clip（包含m个frame）使用CV领域中pretrained模型（如ResNet等）抽取特征向量（visual features），最终视频被表示成特征向量的序列。

从视频中抽取出来的特征向量自然是连续实值向量（属于整个实数空间），和离散的文本有很大的不同。当前，将视频的特征向量注入BERT主要有下面两种方式：

（1）Pipeline方式：将实值向量离散化，和文本token对齐加入到BERT模型中；
（2）端到端的方式：微调BERT模型结构，直接使用实值向量参与计算。

## 《VideoBERT: A Joint Model for Video and Language Representation Learning》

### 1.1 视频文本数据处理（video and language processing）

针对video的处理，首先从input video每秒中抽取20帧画面（20 fps），每30帧组成一个片段。对每个clip用pretrained的ConvNet提取特征向量（1024维）。但是由于特征向量属于整个R^1024空间，是不可数的。为了和文本token相对应，延续原始BERT中的MLM任务，**作者对所有提取出的特征向量使用hierarchical k-means做聚类，一共得到20736个类中心。**把类中心作为visual token，每一个视觉特征向量都由它属于的类中心来表征。
针对文本的处理，使用现成的语音识别工具（Automatic Speech Recognition）提取视频中的文本，利用LSTM-based的语言模型对其断句。后续处理延续原始的BERT，用WordPieces切词，词表大小为3万。

### 1.2 输入格式
![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy81ZmtuYjQxaWI5cUdJN2s4NWRTQmtrVkRmZTRkbE54MTNsUmVDTXpuMVpuNTFsYkNkRE1MMWdMN1VRY2M5azdtNTVmcDdXaWJ1cFB0aWNMRjFkdkhXYXlVQS82NDA?x-oss-process=image/format,png)

### 1.3 自监督任务（pretrain）

原始BERT有两个自监督任务：

（1）cloze（完形填空）/MLM（mask language model）：预测被mask的text token；
（2）NSP（next sentence prediction）：预测两个句对是否是连续的上下句。

第一个任务可以很自然的扩展到visual token中。像text token一样，提前mask visual token，利用没被mask的text token和visual token预测被mask的visual token，是一个多分类问题，使用softmax作为损失函数。

第二个任务NSP在VideoBERT中变成预测text sequence和visual sequence是否一致，即两者是否提取自同一个视频。类似的原始BERT，我们从其他视频数据中抽取visual sequence作为负例，来自该视频数据的visual sequence作为正例。是一个二分类问题。 

### 1.4 下游任务

VideoBERT通过上述两个自监督任务实际上学习了visual-liinguistic的联合表示（分布）p(x,y)，其中x表示visual sequence，y表示text sequence。这个联合分布可以用在下列三种任务上：

（1）text-to-video: 根据文本预测视频，根据文本自动插图。 
       

（2）video-to-text: 根据视频预测文本，对视频自动生成摘要。

（3）unimodal fashion（单一模态下使用）：利用文本或者视频的边缘分布，根据上文预测下文。对文本来说就是我们非常熟悉的语言模型，对于视频来说我们可以根据前面的视频内容预测后面可能发生的事情。
              
## 《Learning Video Representations Using Contrastive Bidirectional Transformer》
将实值连续型的特征向量（visual features）通过聚类规整为有限个类中心，是否会丢失video中包含的很多细节的信息呢(⊙ˍ⊙)？那么，这篇文章就不再使用聚类将实值连续型的visual features离散化，而是直接使用实值向量visual features，通过模型算法上的微调，实现BERT的多模态化。



