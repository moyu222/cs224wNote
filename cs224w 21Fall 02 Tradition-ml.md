---
tags:
  - GNN
  - Stanford
date: 2025-01-02
---
## Traditional Methods for machine Learning in graphs

>[!note] 
>Goal: Make predictions for a set of objects
>Key: Using effective features over graph
>Design choices:
>- 特征：d维度的向量
>- 对象：节点，边，节点的集合，整个图
>- 目标方程：我们要解决的任务

>[!example] 
>Given: $G = (V, E)$ 
>Learn a function : $f: V \to R$ 
>==How do we learn the function?==

## Node-level Tasks and features 节点任务和特征

>[!Overview]
>Goal:Characterize the __the structure and position__ of a node in network
>- 节点度 Node degree
>- 节点中心性 Node centrality
>- 聚合系数 Clustering coefficient
>- Graphlets[^1] (一种描述图结构的单位，可以视作子图，下文详细讲解)

### Node degree
- The degree $k_v$ of node $v$ is the number of edges (neighboring nodes) the node has.
- Treats all neighboring nodes __equally__

### Node Centrality 节点中心性
>[!Overall] 
>节点度没有考虑其邻居节点的重要性importance，节点中心性考虑邻居的重要性。
>
>不同的方法去衡量重要性：
>- Eigenvector centrality
>- Between centrality
>- Closeness centrality
>- and many others...
#### Eigenvector centrality
节点v重要如果他的邻居节点重要
我们将中心性表示为邻居节点的中心性之和，再使用最大特征值normalization
$$
c_{v}= \frac{1}{ \lambda} \sum_{u \in N \left( v \right)}c_{u} \stackrel{ \leftarrow} \rightarrow \lambda c=Ac
$$
将递归关系重写成矩阵形式:
- $\lambda$ is normalization const (largest eigenvalue of A)
- $A$ 为邻接矩阵 
- $c$ : Centrality vector
By Perron-Frobenius Theorem[^2], 最大特征向量总是独特且正的

#### Betweenness centrality
一个节点重要如果他在许多其他节点最短路径上
$$
c_{\upsilon} =\ \sum_{s\neq\upsilon\neq t}\frac{\#\left(\text{shortest paths between s and t that contain v }\right)}{\#\left(\text{shortest paths between s and t }\right)}
$$
![[Pasted image 20241231130730.png]]

### Clustering Coefficient
 Measure __how connected $v's$ neighboring nodes__ are:
 $$
 e_{v}=\frac{\#\left(\text{edges among neighboring nodes}\right)}{\left| \binom{k_{v}}{2}\right|}\in\left[0, 1\right]
 $$
 ![[Pasted image 20241231131129.png]]
 可以简单理解为数节点组成的三角形和所有可能组成的三角形的比例。我们可以延伸这个概念，可以数一些其他图形结构，pre-specified subgraphs, i.e., _graphlets_
 
### Graphlets

_Goals_: 
- 描述节点u周围的网络结构
- Graphlets are small subgraphs that describe the structure of node u's network neighborhood

_Analogy_:
- 度计算边
- 聚类系数数三角
- Graphlet Degree Vector (GDV): Graphlet-base features of nodes
	- GDV 数节点周围的Graphlets

2-5个尺寸的graphlets我们可以得到73维度的向量表示节点周围的拓扑结构，Graphlets提供了nodes's local network topology，比节点度和聚类系数更详细

_Induced Subgraph, Ismorphism_:
- 引导子图：is another graph, formed from a subset of vertices and all of the edges connecting the vertices in that subset.
- 图同构：Two graphs which contain the same number of nodes connected in the same way are said to be isomorphic.

![[Pasted image 20241231133047.png]]
GDV具体例子：
![[Pasted image 20241231133215.png]]
### 总结
>[!summary] 
>我们介绍了多种不同发发获得节点特征，分为两类：
>
>_Importance-based_ features:
>	- Node degree
>	- Diffent node centrality measures
> 用于预测图中有影响的节点：预测社交媒体中的名人用户
> 
>_Stucture-based features_:
>	- Node degree
>	- Clustering coefficient
>	- Graphlet count vector
>用于预测一个节点在图中扮演的特定角色：预测蛋白质功能

### Discussion
![[Pasted image 20241231134311.png]]
目前的节点特征可能只能区分节点标签比较混杂的图，可能无法捕捉到更高结构的特点例如聚类🤔

##  Link Prediction Task and features 链接预测问题和特点
### Recap
- 预测新链接基于现有的链接
- 在测试时，将节点对排列，这些节点对没有链接，前k个节点对构成链接
- 由此可以看出重点在于定义节点的特征
![[Pasted image 20241231135222.png]]
### link prediction task 的两种形式
1. Links missing at random: 随机移除一些链接然后预测他们
2. Links over times: 
	- 使用t0时间的G得到一个链接列表排序然后预测t1时间的link
	- 使用预测正确的link数量
![[Pasted image 20241231135808.png]]

>[!tip] 
>这两个形式应该是可以归纳所有问题，与下文的基于相似性并不冲突，下文讲的是方法论，通过相似性可以预测时间流逝类型，例如，社交网络的朋友关系的产生，有着相同朋友的人更容易成为朋友。或者在推荐系统中，相似的用户可能喜欢相同的事物

### 通过相似性进行预测Link Prediction via Proximity
 对每一对节点(x,y)计算相似度得分c(x,y)，然后进行排序选择top n个作为新链接，然后evaluation
  - $c(x,y)$ 可以是#$x$ 和$y$的共同邻居
  
>[!question] 
>相似性应该属于features，嗯，这里说的就是local neighbood overlap, 所有他为什么这么排版？
### Link-level Features: Overview
- Distance-base feature
- Local neighborhood overlap
- Global neighborhood overlap

### Distance-Based feature
- 即两个节点之间的最短路径距离
![[Pasted image 20241231141902.png]]

### Local Neighborhood Overlap
- 两个点共用的邻居节点
- 分为三种表示
![[Pasted image 20241231142019.png]]

### Global Neighborhood Overlap --Katz index

>[!tip] 
>local方法在两个节点没有共同邻居时失效，但是两个节点在在未来有可能建立联系。

Katz index: 计算给定节点之间所有长度的游走数量，==_利用邻接矩阵的幂_==
邻接矩阵可以看着长度为1的游走链接情况，每一次幂可以看作多走长度1
![[Pasted image 20241231142952.png]]
![[Pasted image 20241231143334.png]]
Katz index 计算：
![[Pasted image 20241231143451.png]]
具体计算涉及数学，看懂就行。

>[!summary] 
>
![[Pasted image 20241231143829.png]]

## Graph-Level Features and Graph Kenels
- Goals: 捕捉整个图的结构

### Background : Kernel Method 核方法
![[Pasted image 20241231144520.png]]
>[!tip] 
>核好像就是一种可以在高维特征空间描述相似性的东西，在SVM中有很多应用，感觉类似hash函数就是一个得到特征的方法，有关键性质可以表示为内积，即存在一个映射 $\phi$ 使得：$K(x,y)=ϕ(x)⊤ϕ(y)$
>

### Graph-level Features: Overview
图核：测量两个图的相似性
- Graphlet Kernel
- Weisfeiler-Lehman Kernel
...

### Graph Kernel: Key Idea - Bag of *
![[Pasted image 20241231150302.png]]
>[!tip] 
此处讲述的特征抽取方法也将是bag-of-something的形式，将图表示成一个向量，每个元素代表对应something出现的次数（这个something可以是node, degree, graphlet, color），图片中的就是bag of node 将节点数量映射到某个值，但是这明显不足以计算相似度，我们可以用degree或者graphlets

![[Pasted image 20241231150634.png]]

### Graphlet features
- Key idea: Count the number of different graphlets in a graph.
- 注意这里对graphlet的定义跟上文节点层面特征抽取里的graphlet不一样。区别在于：
	Nodes in graphlets here do not need to be connected (allows for isolated nodes)
	The graphlets here are not rooted.
- 对每一种节点数，可选的graphlet：
![[Pasted image 20241231150918.png]]
Given graph G, and a graphlet list $G_{k}=$   $\left( g_{1,}g_{2},...,g_{n_{k}} \right),$ define the graphlet count  $vectorf_{G} \in{R}^{n_{k}}as$  

$$\left( f_{G} \right)_{i}= \# \left( g_{i} \subseteq G \right) for i=1,2,... ,n_{k}$$

![[Pasted image 20241231151258.png]]
根据核的定义我们计算出graphlet kernel,但是因为不同图有不同大小，我们需要normalize
$$K \left( G,G^{ \prime} \right)=f_{G}^{T}f_{G^{ \prime}}$$
$${h}_{G}=\frac{ {f}_{G}}{ Sum( { f}_{G})}\qquad K( {G}, {G}^{\prime})= {h}_{G} {}^{\rm T} {h}_{G^{\prime}}$$

### Limitation
![[Pasted image 20241231151721.png]]

### Weisfeiler-Lehman KERNEL 
- 相比graphlet kernel代价较小，效率更高。 
- 用节点邻居结构迭代地来扩充节点信息，使用bag of degrees
- 使用color refinement 算法，就是迭代上色
![[Pasted image 20241231151832.png]]
### Color Refinement
- 分配初始颜色
- 使用hash函数迭代颜色
Given: A graph $G$ with a set of nodes V. 
- Assign an initial color c(o) (v) to each node v. 
- Iteratively refine node colors by ..where HASH maps different inputs to different colors.

$$c^{ \left( k+1 \right)} \left( v \right)=HASH \left( \left \{ c^{ \left( k \right)} \left( v \right), \left \{ c^{ \left( k \right)} \left( u \right) \right \}_{u \in N \left( v \right)} \right \} \right),$$

- After $K$ steps of color refinement, $c^{(K)}$ (v) summarizes the structure of K-hop neighborhood
![[Pasted image 20241231152338.png]]
![[Pasted image 20241231152407.png]]
![[Pasted image 20241231152413.png]]
![[Pasted image 20241231152427.png]]
### Weisfeier-Lehman Graph features
![[Pasted image 20241231152506.png]]
![[Pasted image 20241231152519.png]]
>[!summary] 
>![[Pasted image 20241231152618.png]]

## Today summary
![[Pasted image 20241231152644.png]]



# Footnotes

[^1]: Graphlet 是一种表示数据管道部分的结构，通常以有向图的形式描述。可以将其视为一个图的子集，其中不仅包含节点，还可以包含其他图形。
[^2]: 对于一个非负的方阵，所有元素不小于0，存在最大特征值非负且绝对值最大，存在与之对应的非负特征向量