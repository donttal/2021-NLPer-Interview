# 基于Transformers+CNN/LSTM/GRU的文本分类
[参考网址](https://mp.weixin.qq.com/s/zSuKPVgeneD1GpFOIiOJ2A)
[github地址](https://github.com/zhanlaoban/Transformers_for_Text_Classification)
**基于最新的 huggingface 出品的 transformers v2.2.2代码进行重构。为了保证代码日后可以直接复现而不出现兼容性问题，这里将 transformers 放在本地进行调用。**
## Highlights
支持transformer模型后接各种特征提取器
支持测试集预测代码
精简原始transformers代码，使之更适合文本分类任务
优化logging终端输出，使之输出内容更加合理
Support

### model_type
[✔] bert
[✔] bert+cnn
[✔] bert+lstm
[✔] bert+gru
[✔] xlnet
[ ] xlnet+cnn
[✔] xlnet+lstm
[✔] xlnet+gru
[ ] albert

### Content
- dataset：存放数据集
- pretrained_models：存放预训练模型
- transformers：transformers文件夹
- results：存放训练结果
- 
## Usage
### 1. 使用不同模型
在shell文件中修改model_type参数即可指定模型
如，BERT后接FC全连接层，则直接设置model_type=bert；BERT后接CNN卷积层，则设置model_type=bert_cnn.
在本README的Support中列出了本项目中各个预训练模型支持的model_type。
最后，在终端直接运行shell文件即可，如：
bash run_classifier.sh
注：在中文RoBERTa、ERNIE、BERT_wwm这三种预训练语言模型中，均使用BERT的model_type进行加载。
### 2. 使用自定义数据集
在dataset文件夹里存放自定义的数据集文件夹，如TestData.
在根目录下的utils.py中，仿照class THUNewsProcessor写一个自己的类，如命名为class TestDataProcessor，并在tasks_num_labels, processors, output_modes三个dict中添加相应内容.
最后，在你需要运行的shell文件中修改TASK_NAME为你的任务名称，如TestData.
### Environment
- one 2080Ti, 12GB RAM
- Python: 3.6.5
- PyTorch: 1.3.1
- TensorFlow: 1.14.0(仅为了支持TensorBoard，无其他作用)
- Numpy: 1.14.6
- 
## Performance
数据集: THUNews/5_5000
epoch:1
train_steps: 5000
![](https://cdn.jsdelivr.net/gh/donttal/figurebed/img/文本分类效果图.png)

## Download Chinese Pre-trained Models
NPL_PEMDC(https://github.com/zhanlaoban/NLP_PEMDC)