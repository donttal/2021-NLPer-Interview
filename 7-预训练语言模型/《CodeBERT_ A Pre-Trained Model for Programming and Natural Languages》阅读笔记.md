# 《CodeBERT: A Pre-Trained Model for Programming and Natural Languages》阅读笔记
[参考网址](https://zhuanlan.zhihu.com/p/108441952)
CodeBERT 学习能够支持下游 NL-PL 应用的通用表示，比如自然语言代码搜索、代码文档生成，经实验 CodeBERT 模型在两项任务均取得 SOTA 效果，同时研究者构建了 NL-PL 探测数据集，CodeBERT 在 zero-shot 设置中的性能表现也持续优于 RoBERTa。

CodeBERT 模型使用基于 Transformer 的神经架构构建而成，训练所用的混合目标函数包括了替换 token 检测（replaced token detection，RTD）预训练任务。RTD 使用从生成器采样的合理替代 token 来替换部分输入 token 从而破坏输入，然后训练一个判别器来预测受损输入中的每个 token 是否被生成器样本替换。

这就使得 CodeBERT 模型可利用双模态数据 NL-PL 对和单模态数据，前者为模型训练提供输入 token，后者有助于学得更好的生成器，研究者通过模型调参的方式评估了 CodeBERT 在两个 NL-PL 应用中的性能。