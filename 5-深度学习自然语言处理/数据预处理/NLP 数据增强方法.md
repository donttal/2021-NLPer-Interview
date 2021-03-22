# NLP 数据增强方法

https://zhuanlan.zhihu.com/p/75207641

https://www.qbitai.com/2020/06/16103.html

https://www.dataapplab.com/enhance-nlp-what-are-the-easiest-use-augmentation-techniques/

- 同义词替代：
- 词嵌入替换： 采用嵌入空间中最近的邻接词作为句子中某些单词的替换
- 掩码语言模型：通过bert等这种MLM模型来预测被mask掉的词来做替换，需要注意的是决定哪一个单词被mask是比较重要的
- 基于TF-IDF的单词替换
- 文本形式转换 利用正则表达式应用的的简单模式匹配转换
- 回译 我们用机器翻译把一段中文翻译成另一种语言，然后再翻译回中文。
- 随机噪声注入 
- 语法树结构替换
- 篇章截取
- seq2seq序列生成数据
- 生成对抗网络 GAN
- 预训练的语言模型