## 题目背景
[网址](https://zhuanlan.zhihu.com/p/344299762)
在这个竞赛中，你的挑战是创建“知识追踪”的算法，即随着时间推移对学生知识的建模。目标是准确预测学生在未来互动中的表现。您将使用Riiid的EdNet数据对您的机器学习技能进行配对。

你的创新算法将有助于应对全球教育挑战。如果成功的话，任何有互联网连接的学生都有可能享受到个性化学习体验的好处，而不管他们住在哪里。有了你们的参与，我们可以在后COVID-19世界建立一个更好、更公平的教育模式。


# SAINT 和 SAINT+
[网址](https://zhuanlan.zhihu.com/p/351250371)
SAINT系列是Transformer-Based的模型
首先它是一个Transformer-Based的模型，很适合处理序列问题。相比SAKT[1]模型，多了时间特征（Elapsed Time和Lag Time）和Decoder。和RNN相比运算更快。

![](https://pic3.zhimg.com/v2-4e3929d73e5632b9f71ebbf59fb7ed4e_r.jpg)

根据SAINT+的paper我们复现了SAINT+的网络，这个网络既有Encoder也有Decoder，需要注意一点是在**Encoder也需要使用上三角的Mask，避免信息泄露。**

时间特征对精度提升十分重要，时间特征有两个，分别是Elapsed Time和Lag time。Elapsed Time是指用户完成一个Exercise所花费的时间；Lag Time是指用户在进行下一个Exercise的时候距离上一个Exercise所间隔的时间。用数学表达式就是：

$$
\begin{array}{l}
\text { ElapsedTime }: E 0, E 1, \ldots E n \in[0,300] s \\
\text { LagTime : } L 0=T 1-(T 0+E 0), L 1=T 2-(T 1+E 1), \ldots, L n=T n+1-(T n+E n)
\end{array}
$$

## 特征选择
特征的选择大体参照了SAINT系列的paper，只是略微有些不同。

![](https://pic4.zhimg.com/v2-329b4375a5ede4ee1dadc7ade4ba224b_r.jpg)

![](https://pic2.zhimg.com/v2-5fcc7597802e9b68d7f7e58dbcc95595_r.jpg)

Exercise id：问题或者课程的编号

Position：位置编码

Part：问题或者课程属于的Part

Correctness：上个问题回答正确与否

Elapsed time：回答上个问题的耗时

Lag time：两个活动之间的间隔时间

Prior had explanation：上个问题是否有解释



## Ensemble
最后用了两个SAINT+模型和一个LGBM做ensemble。通过多次测试，找最优的系数。


## 缺点
目前应用在知识跟踪上的注意力机制有如下两个限制，如下图：
（1）以前模型的注意力层太浅，不能捕捉不同练习题（exercise）和答案（response）之间复杂的关系。
（2）合适的构建queries, keys and values还没有被充分发掘。比如目前都是练习题为queries，交互为keys和values。