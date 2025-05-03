---
tags:
  - Stanford
  - GNN
date: 2025-01-02
---
## Node embeddings

### Recap
- 图表示学习是一种技术，旨在将图中的节点、边或整个图转化为低维向量表示。这种表示可以捕捉图的结构和属性特征，便于后续的机器学习任务（如分类、聚类等）。避免了传统机器学习每次都要进行的特征工程
- 节点嵌入式图表示学习的一种特定形式
![[Pasted image 20250102134230.png]]

### Why embedding?
- 相似的嵌入对应着节点在网络中相似性
- 对网络信息编码
- 用于下游多种预测
![[Pasted image 20250102134507.png]]

### Example
![[Pasted image 20250102134532.png]]

## Node Embeddings: Encoder and Decoder

### Set up
- $V$ is the vertex set.
- $A$ is the adjacency matrix (assume binary)
- No node features or extra information

### Embedding nodes
- 对节点进行编码，例如点积dot product 近似图中的相似性。
>[!tip] 
>点积似乎是对应着两个向量之间的cos值
>嵌入过程需要encoder 和 decoder
>下图中的encoder就是ENC(u),编码器就是一个函数，映射
>我们使用decoder来定义相似性，$Z_{v}^{T}Z_{u}$就是decoder，结果表示similarity

![[Pasted image 20250102135120.png]]
- Encoder maps from nodes to embeddings 映射到低维向量
- Decoder DEC caps from embeddings to similarity score 
- ==Optimize the parameters of the encoder==

### “Shallow" Encoding
- 最简单的encode方法：Encoder只是一个嵌入查找器，==直接优化嵌入值==。
- 通过one-hop的 $v$ 来查找
![[Pasted image 20250102140127.png]]
![[Pasted image 20250102140245.png]]
Many methods; Deepwalk, node2vec

### Framework summary
![[Pasted image 20250102140404.png]]

### How to Define Node Similarity
- 方法的关键是如何定义相似性，上述的点积是一种方法
- emample：
	- linked?
	- share neighbors?
	- have similar "structural roles?"
- ==我们现在学习使用随机游走random walks定义相似性并优化嵌入==

>[!note] 
>- 这是无监督或者说自监督方法学习节点嵌入，不使用节点标签或者特征
>- 这些嵌入与任务无关，不是为了特定的任务但是可以被任意任务使用

## Random Walk Approaches For Node embeddings

### Notation 符号
- Vector $z _{u}$ ：The embedding of node $u$ (what we aim to find). 
- Probability $P \left( v|z_{u} \right)$ :       $\gets$ Our model prediction based on $z_{u}$
	- 从节点u随机游走到v的概率
---
我们使用非线性的函数得到predicted probabilities
- Softmax function:
	- 将模型预测得到的k个值，转化成和为一的概率
	- $\sigma \left( z \right) \left[ i \right]= \frac{e^{z \left[ i \right]}}{ \sum_{j=1}^{K}e^{z \left[ j \right]}}$
- Sigmoid fuction:
	- 将得到的值转换成0到1之间
	- $S \left( x \right)= \frac{1}{1+e^{-x}}$

![[Pasted image 20250102142522.png]]

### Random-Walk Embeddings
![[Pasted image 20250102142550.png]]
![[Pasted image 20250102142648.png]]

### Why Ramdom Walk?
- Epressivity：
    - 它结合了**局部**（直接邻居）和**高阶**邻域信息（可能相隔多跳的节点）。
    - **关键思想**：如果从节点 u 开始的随机游走以高概率访问节点 v，则认为 u 和 v 是相似的。这反映了通过多跳路径连接的高概率性。
- Efficiency：
    - 它只需关注在随机游走中共同出现的节点对。这显著减少了计算负担。

### Feature Learning as Optimization
![[Pasted image 20250102143221.png]]

>[!tip] 
>- 对于给定的节点 u，我们的目的是学习特征表示（嵌入），这些表示能够有效地预测其随机游走邻域内节点的属性或行为，这个邻域用 NR(u)表示。
>- ==这个邻域可能包括与 u 直接相连的节点，以及通过多次跳转可以到达的节点。==
>- 节点邻域的访问概率应该是很高的，我们通过这个条件，将嵌入问题转化成优化问题
>- Log-likelihood objective[^1]

### Random Walk Optimization
1. 进行短的固定长度的随机游走，从u使用随机游走策略R
2. 对于每个u得到multiset即$N_R(u)$
3. 优化

![[Pasted image 20250102144555.png]]
![[Pasted image 20250102144617.png]]
![[Pasted image 20250102144842.png]]
>[!tip] 
>因为使用softmax，相似性值(0,1)，我们取负号使得L最小
>归一化即分母导致成本昂贵

### Negative Sampling
![[Pasted image 20250102145126.png]]
>[!tip] 
>负样本是指不与目标节点 u 直接相连的节点，或者在从 u 出发的随机游走中未被访问的节点
>- 负样本的概率与其度（连接的边数）成正比
>- 对于 k 的考虑
>	1. 选择更多的负样本（即更大的 k）可以使模型对负样本的学习更全面
>	2. 较大的 k 可能导致模型对负事件的偏差增加，这意味着模型可能会更倾向于将某些节点标记为负样本，而忽略一些潜在的正样本。在实际应用中，通常选择 k 的值在 5 到 20 之间。

### Stochastic Gradient Descent SDG 随机梯度下降
![[Pasted image 20250102150552.png]]

>[!summary] Random Walks Summary
>![[Pasted image 20250102150619.png]]

### How should we randomly walk?
![[Pasted image 20250102150751.png]]
引出下一个方法node2vec 

### Overview of node2vec 
- Goals: 同上
- ==使用Maxium likehood==
- 灵活地定义网络邻域$N_Ru$可以捕捉到节点 u 的多层次、多样化的连接关系，这有助于生成更丰富的节点嵌入。就是更强了
- ==biased 2nd order random walk 𝑅==

### node2vec: Biased Walks
![[Pasted image 20250102151332.png]]
Idea:
使用灵活偏置随机游走权衡local和global视角。[论文](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf)

![[Pasted image 20250102152006.png]]
![[Pasted image 20250102152034.png]]

>[!tip] 
>两个参数p和q分别代表回到前一个节点和BFS与DFS的比例
>这种方法应该比上一个deepwalk花费更少的游走次数

![[Pasted image 20250102152706.png]]
![[Pasted image 20250102152959.png]]
>[!question] 
>为什么较低的p值是BFS?
>不知道，先默认返回概率低就对于更倾向于bfs

### node2vec algorithm 
![[Pasted image 20250102153653.png]]
- 优化过程每个节点可以并行，嗯不到为什么，上一个不能吗
- 线型的时间复杂度

### Other idea 
![[Pasted image 20250102153858.png]]

>[!summary] Summary so far 
>![[Pasted image 20250102153951.png]]
>![[Pasted image 20250102154036.png]]

## Embedding Entire Graphs

![[Pasted image 20250102161155.png]]

### Approach 1
简单有效的方法1:
- 对每个节点嵌入求和平均
- $z_{G}= \sum_{v \in G}z_{v}$

### Approach 2
引入一个虚拟节点来代表整个图或者子图，然后运行标准的图嵌入
![[Pasted image 20250102161736.png]]

### Approch 3 Anonymous Walk Embedding 匿名游走嵌入
匿名漫步中的状态对应于我们在随机漫步中第一次访问节点的索引，以节点第一次出现的序号（是第几个出现的节点）作为索引
![[Pasted image 20250102162101.png]]
![[Pasted image 20250102162345.png]]
![[Pasted image 20250102162353.png]]
>[!tip] 
>- 随机游走可能往复a,b,a,b等等，匿名漫步对应固定长度所有可能的节点序列状况
>- 匿名漫步的数量指数级上升
>- 有些项Graphlets，和GDV

### simple use of anonymous walk
![[Pasted image 20250102162751.png]]

### Sampling Anonymous Walks
- 对匿名游走进行采样：生成独立的m个随机游走的集合
- 将图表示为这些游走的概率分布
- m的数量：
	- ![[Pasted image 20250102163046.png]]
那个0一样的东西感觉是置信度之类的东西
### New idea: Learn Walk embeddings
上文我们使用出现的次数简单地代表每个游走，我们现在学习如何将匿名游走嵌入
![[Pasted image 20250102163407.png]]
![[Pasted image 20250102165019.png]]
![[Pasted image 20250102165028.png]]

>[!tip] 
>- 注意第二，三张图的公式错了，应该是从$t-\vartriangle$ 到$t+\vartriangle$，可能也没错？反正就是通过他们预测目标游走
>- 我们可以简单理解游走嵌入和上文的deepwalk思想相同。例如从节点1开始相同长度的匿名游走更可能同时出现，我们将上文有节点组成的邻域转化成相同长度匿名游走组成的邻域，使用$\vartriangle$限定范围，看作一个小窗
>- 具体看论文吧我也不懂

![[Pasted image 20250102165923.png]]
得到图的嵌入后我们可以使用第二讲的核方法或者NN进行分类

>[!summary] 
>![[Pasted image 20250102170102.png]]

### Preveiw: Hierarchical Embedding 分层嵌入
![[Pasted image 20250102170154.png]]
后面再说。
>[!tip] 
>来了，来了，[[cs224w 21Fall 08 Applications of GNN#Hierarchical Global Pooling 分层嵌入]]，基础的sum或avg方法可能会丢失信息

### How to use embeddings
![[Pasted image 20250102170250.png]]
好多下游任务

## Today summary
![[Pasted image 20250102170328.png]]


# Footnotes

[^1]: 最大似然估计，我们认为现实发生的情况就是最有可能发生的情况，所以在未知参数的情况下写出发生现实情况的概率公式，认为使其最大的参数就是我们所需的参数。所以最大化这个公式，就能得到我们要的参数。  因为对连乘取log会使其更便于计算，所以一般会对其取log。具体以后再说