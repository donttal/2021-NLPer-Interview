# RoBERT更大数据规模和仔细调参的最优BERT
文章标题：RoBERTa: A Robustly Optimized BERT Pretraining Approach

文章作者：Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov

文章链接：https://arxiv.org/pdf/1907.11692.pdf

代码链接：https://github.com/pytorch/fairseq

## 导言
这篇论文将炼丹炉火力全开，在得到以下炼丹配方后将BERT的效果再次提到SOTA：
- 训练更久，batch更大，数据更多
- 丢弃NSP
- 在更长的序列上训练
- 动态改变mask策略

在新的炼丹配方上训练，RoBERT在GLUE上达到了88.5，在4/9上的任务上刷新SOTA，且在SQuAD和RACE上更新记录。

总的来说，本论文的贡献在于：

- 修改了BERT原有的训练策略
- 引入了一个更大的新数据集CCNEWS
- 验证了预训练模型合理设计的重要性

## 新的调整
- 对学习率的峰值和warm-up更新步数作出调整（在不同任务下不同，在实验阶段说明）
- 将Adam中$\beta_{2}=0.999$改为$\beta_{2}=0.98$
- 不对序列进行截短，使用全长度序列
- 使用更高级的炼丹炉DGX-1 each with 8*32GB NVIDIA V100 GPUs interconnected by Infiniband
- 使用更多的炼丹原料，共160G左右，包括：
**原BERT的16GB原料**
CC-NEWS的76GB原料
OPENWEBTEXT的38GB原料
STORIES的31GB原料

### 动态Mask
动态mask就是训一个mask一个，这样基本可以保证每次看到的都不一样
![](https://pic3.zhimg.com/80/v2-f598af1059be6653a8667dcec01d8ece_hd.jpg)

### 输入格式和NSP
原BERT使用了NSP，但是近期的一些文章质疑了NSP的作用，并且是把两个句子贴在一起喂给模型，本文讨论了下面的组合：

- SEGMENT-PAIR+NSP。这就是原BERT的NSP方法，输入的是两个段，每个段都可能有多个句子，总长度要小于512。
- SENTENCE-PAIR+NSP。这里从各自的文档或段中sample一个句子即可，因为总长度小于512，所以增大了batch size使得总字符数和SEGMENT-PAIR差不多。
- FULL-SENTENCES。直接从文档中连续sample句子直到塞满512的长度，且丢掉NSP。当到达一个文档结尾时，先增加一个文档分隔符，再从后面的文档接着sample。
- DOC-SENTENCES。和上面一样，只是不得(may not)跨文档。同样增大batch size 使得总字符数和上面差不多。

![](https://pic2.zhimg.com/80/v2-d4a914efb544acc18a03540eb00f553d_hd.jpg)

首先看最上面两行，使用多个句子效果更好，因为可以捕捉句子之间的相关性。再看中间两行，差不多，但是注意到都好于有NSP的模型，这说明NSP的确没用。而且似乎DOC-SENTENCES比FULL-SENTENCES略好，但这难以控制batch size，所以后面我们固定使用FULL-SENTENCES。

### 更大的batch size
原BERT使用了bs=256和1M步的训练方法，这等价于bs=2K和125K步或者bs=8K和31K步，下面是在BOOKCORPUS和WIKIPEDIA上的实验结果
![](https://pic3.zhimg.com/80/v2-35d81d711433f07cea4cd30aac35dbe6_hd.jpg)
好像趋势不太明显，只能说总的来看bs更大效果更好。

## 原子单位
时下流行的方法是BPE，传统上我们使用unicode（两个byte）作为切分基础，Radford et al. (2019) 使用了 byte-level 方法将字典大小限制在了50K左右，所以本文使用这种方法，且不对数据进行任何预处理。

## 实验
总结一下上面，RoBERT作了如下调整：

使用动态mask
- FULL-SENTENCES without NSP
- 更大的bs
- 更大的byte-level BPE
此外，还有两个重要的因素有待探究：

- 数据量大小
- 训练次数
所以，我们首先复现BERT_large作为基线模型，包括原来的数据集，然后在相同的结构和数据集上实现RoBERT，然后再在160G的新数据集上训练，且增加训练步数。下表是结果：
![](https://pic4.zhimg.com/80/v2-696ee1204c15520ae7f370e817c14a9b_hd.jpg)
结果是显然的，更多的数据和更长的训练会提高效果，这告诉我们一个道理：大力出奇迹。

### GLUE
对于GLUE，我们考虑两种设置

对不同的任务各自微调，在bs[公式]和lr[公式]之间选择，并且对前6%的训练步数进行线性warm-up，然后使用线性衰减。此外，用10个epoch微调与提前终止，其他保持前文所述不变。
和GLUE榜上的其他模型一较高下，但是只基于单任务微调。对RTE，STS和MRPC使用MNLI单任务微调。对于QNLI和WNLI这两个任务有独特的微调方法，详情参考论文及附录。下表是结果：
![](https://pic3.zhimg.com/80/v2-b262be502fd936346d38e06898c7eb5a_hd.jpg)

### SQuAD
下表是SQuAD1/2的结果：
![](https://pic2.zhimg.com/80/v2-c0481e1ce54d0ca06b6559bf48e2fe55_hd.jpg)
其中，带有† 表示引入了额外的外部数据。可以看到，即使在有外部数据的情况下，RoBERT也比BERT_large好。

### RACE
![](https://pic1.zhimg.com/80/v2-28e56502b64f612225eb7e3e22628bd4_hd.jpg)