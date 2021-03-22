# BERT实战
## BERT_Readme文件
1. 新技术：新技术称为全字掩蔽。在这种情况下，我们总是会同时屏蔽 与一个单词相对应的所有标记。总体掩盖率保持不变。Whole Word Masked Input: the man [MASK] up , put his basket on [MASK] [MASK] [MASK] ' s head
2. BERT已上传到TensorFlow Hub。[参考网址](https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb)
3. BERT预训练模型以zip模式，每个.zip文件包含三个项目：

* 一个TensorFlow检查点（bert_model.ckpt），其中包含预先训练的权重（实际上是3个文件）。
* vocab文件（vocab.txt），用于将WordPiece映射到单词ID。
* 一个配置文件（bert_config.json），用于指定模型的超参数。

1. BERT-Base使用给定的超参数，使用的微调示例应能够在具有至少12GB RAM的GPU上运行。

## Bert代码结构
![](https://pic3.zhimg.com/80/v2-a77b534846c87fc99816abc42481539e_hd.jpg)

## 使用BERT微调任务
[参考网址](https://github.com/google-research/bert)
1. 句子（和句子对）分类任务
2. SQuAD1.1
3. SQuAD2.0

## 微雕模型run_classifier.py和run_squad.py的create _model 部分核心代码
![](https://pic3.zhimg.com/80/v2-925c42991bd202c937d167c1319937ee_hd.jpg)
![](https://pic1.zhimg.com/80/v2-1bb58f9981cade50f27b5ed91bcd7f5c_hd.jpg)
从代码中可以看到，run_squad.py和run_classifier.py微调模型是一层简单的全链接层，以此类推，如果你要实现命名实体识别等其他目标任务，可在预训练的模型基础上，加入少了全链接层。

### 内存不足
原项目所有实验均在具有64GB设备RAM的Cloud TPU上进行了微调。因此，当使用具有12GB-16GB RAM的GPU时，如果使用与本文所述相同的超参数，则可能会遇到内存不足的问题。
**影响内存使用量的因素有：**
**max_seq_length：**发布的模型经过训练，序列长度最大为512，但是您可以使用更短的最大序列长度进行微调，以节省大量内存。这由max_seq_length示例代码中的标志控制。

**train_batch_size：**内存使用也与批处理大小成正比。

**模型类型，**BERT-Base vs .BERT-Large：BERT-Large模型比所需的内存更多BERT-Base。

**优化程序：**BERT的默认优化程序是Adam，它需要大量额外的内存来存储m和v向量。切换到内存效率更高的优化器可以减少内存使用量，但也会影响结果。我们尚未尝试使用其他优化器进行微调。
**作者建议**
系统	序列长度	最大批量
BERT-Base	64	64
...	128	32
...	256	16
...	320	14
...	384	12
...	512	6
BERT-Large	64	12
...	128	6
...	256	2
...	320	1个
...	384	0
...	512	0

## 使用BERT提取固定特征向量
在某些情况下，与其从头到尾微调整个预先训练的模型，不如获取预先训练的上下文嵌入，这是有益的，这些嵌入是从预先隐藏的隐藏层生成的每个输入令牌的固定上下文表示，这可能是有益的。训练的模型。这也应减轻大多数内存不足的问题。

## Tokenization
对于句子级任务（或句子对）任务，标记化非常简单。 只需遵run_classifier.py和extract_features.py中的示例代码即可。 
句子级任务的基本过程是：
1. Instantiate an instance of `tokenizer = tokenization.FullTokenizer`
2. Tokenize the raw text with `tokens = tokenizer.tokenize(raw_text)`.
3. Truncate to the maximum sequence length. (You can use up to 512, but you probably want to use shorter if possible for memory and speed reasons.)
4. Add the `[CLS]` and `[SEP]` tokens in the right place.
在描述处理单词级任务的一般方法之前，重要的是要了解我们的令牌生成器到底在做什么。它包含三个主要步骤：

文本规范化：将所有空格字符转换为空格，并（对于Uncased模型）将输入小写并去除重音标记。例如John Johanson's, → john johanson's,。

标点符号拆分：在两侧拆分所有标点符号（即，在所有标点符号周围添加空格）。标点符号定义为（a）带有P*Unicode类的所有字符，（b）任何非字母/数字/空格ASCII字符（例如，$从技术上讲不是标点符号的字符）。例如，john johanson's, → john johanson ' s ,

WordPiece令牌化：将空格令牌化应用于上述过程的输出，并将 WordPiece 令牌化分别应用于每个令牌。（我们的实现直接基于tensor2tensor链接的）。例如，john johanson ' s , → john johan ##son ' s ,

## BERT预训练
将发布代码以对任意文本语料库进行“掩盖LM”和“下一句预测”。

## 前期准备
在 https://github.com/google-research/bert 下载模型的源码包。
在https://github.com/google-research/bert 下方下载我们需要的预训练模型文件，
我们选择中文的，下载下来是一个zip chinese_L-12_H-768_A-12.zip
最后准备我们自己的训练数据。讲上述数据都一并放到源码包目录下。

## 配置代码
sh脚本文件进行配置
```
python run_classifier.py \
  --data_dir=./dataset \
  --task_name=youself \
  --vocab_file=./chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json \
  --output_dir=./output/ \
  --do_train=true \
  --do_predict=true \
  --do_eval=false \
  --init_checkpoint=./chinese_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=200 \
  --train_batch_size=16 \
  --learning_rate=5e-5 \
  --num_train_epochs=5.0
```
主要的是：

1、指定预训练模型的路径和配置文件路径也就是刚才下载的chinese_L-12_H-768_A-12.zip

2、指定一些do_train 和 do_predict 等等，按照你需要的配置

3、指定你自己的训练数据，这边的训练数据的格式是有规定的，源码中的是cvs，文本在前面标签在后面。

## 修改输入数据格式
源码的数据格式
```
class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines
```
自己定义类
```
class YouSelProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_examples(
        self._read_txt(os.path.join(data_dir, "train.txt")), "train")
 
  def get_dev_examples(self, data_dir):
    return self._create_examples(
        self._read_txt(os.path.join(data_dir, "dev_matched.txt")),
        "dev_matched")
 
  def get_test_examples(self, data_dir):
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_matched.txt")), "test")
 
  def get_labels(self):
    return ["1", "2", "3"]
 
  def _create_examples(self, lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
        line = convertYouselfLine(line)
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples
```
再在main函数中添加
```
processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mrpc": MrpcProcessor,
    "xnli": XnliProcessor,
    "youself": YouSelProcessor
}
```

# bert迁移的实现过程
```
if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:
 
        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()
 
        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
```
init_checkpoint 就是我们指定与训练模型的位置，chinese_L-12_H-768_A-12。

modeling.get_assignment_map_from_checkpoint，函数会检查以及加载chinese_L-12_H-768_A-12模型里的变量和参数形成一个map。

然后tf.train.init_from_checkpoint对这个里面的参数进行初始化。

也就是说在你运行的时候，model里面的变量，通过上面的步骤就已经模块化了一次了，接下来你再次训练，就是从上次的过程中继续往下，或者你也可以说直接不训练了，在以前的模型上直接使用