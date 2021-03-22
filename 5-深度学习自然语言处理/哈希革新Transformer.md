# 哈希革新Transformer
[知乎专文](https://zhuanlan.zhihu.com/p/99737208)
**文章介绍了两种提高 Transformer 效率的技术，最终的 Reformer 模型和 Transformer 模型在性能上表现相似，并且在长序列中拥有更高的存储效率和更快的速度。**

  **在最开始，文章提出了将点乘注意力（dot-product attention）替换为一个使用局部敏感哈希（locality-sensitive hashing）的点乘注意力，将复杂度从 O($L^2$) 变为 O($L log L$)，此处 L 指序列的长度。**
  
 **此外，研究者使用可逆残差（reversible residual layers）代替标准残差（standard residuals），这使得存储在训练过程中仅激活一次，而不是 n 次（此处 n 指层数）。最终的 Reformer 模型和 Transformer 模型在性能上表现相同，同时在长序列中拥有更高的存储效率和更快的速度。**
 
[论文地址](https://openreview.net/forum?id=rkgNKkHtvB)
[代码](https://github.com/google/trax/blob/master/trax/models/research/reformer.py)

## Transformer 上的内存占用问题
- 由于激活需要被存储并用于反向传播，有着 N 层的模型的大小比单层大了 N 倍；
- 由于中间的全连接层的深度 d_ff 通常远大于注意力激活层的深度 d_model，因此需要占用很大的内存；
- 在长度为 L 的序列上的 attention 的计算和时间复杂度是O($L^2$)，所以即使是一个有 64K 字符的序列就会耗尽 GPU 的内存。

## 解决方法
- 可逆层（Reversible layer），这个东西最早是 Gomez 等人引入的，在整个模型中启用单个副本，所以 N factor 就消失了；
- 在前馈层（feed-forward layer）分开激活和分块处理，消除 d_ff factor，节省前馈层的内存；
- 基于局部敏感哈希（locality-sensitive hashing，LSH）的近似注意力计算，让注意力层的 O(L2) 因子替代 O(L) 因子，实现在长序列上的操作。

