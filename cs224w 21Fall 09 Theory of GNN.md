---
date: 2025-01-08
tags:
  - GNN
  - Stanford
---
## How Expressive are Graph Neural network

### Recap GNN 
![[Pasted image 20250108162049.png]]

### Idea: Aggregate neighbors
- key idea: 基于局部网络邻域生成节点嵌入
- Intuition：节点使用神经网络从他的邻居中聚合节点
![[Pasted image 20250108162311.png]]

### Theory of GNNs 
>[!tip] 
>- 本节的GNN理论应该就是指模型的表现能力expressive power ，即区分不同图结构的能力
>- 因此我们有两个问题：
>	- how powerful are gnn 即定义expressive Power
>	- how to design a maximally expressive GNN model 

### Backgroud 

#### Many GNN Models 
![[Pasted image 20250108162955.png]]
- GCN (mean-pool) : element-wise mean pooling + Linear + ReLU non-linearity[[cs224w 21Fall 07 GNN2 具体#Classical GNN layers GCN]]
- GraphSAGE (max-pool) : MLP + element-wise max-pooling [[cs224w 21Fall 07 GNN2 具体#Classical GNN layers GraphSAGE]]

#### Node color/features
![[Pasted image 20250108163728.png]]

#### Local Neighborhood Structures 
1. specigically consider ==local neighborhood structures== around each node in a graph 
	- different node degrees
	- their neighbors have different node degrees 
	- 相同如果他们在图中对称
2. key question: 节点嵌入是否能区分局部邻域结构 ，我们延伸到另一个关键概念计算图
3. ==key：不同邻域结构定义不同计算图定义不同节点嵌入==

#### Computational Graph 计算图
1. GNN通过计算图生成节点嵌入
![[Pasted image 20250108170355.png]]
2. 在上文对节点只定义了color因此只能捕捉节点特征，因此节点1，2生成相同的嵌入。上文使用特征增广进行解决[[cs224w 21Fall 08 Applications of GNN#某些结构GNN很难学习]]
3. 计算图对应rooted subtree structure，通过从根节点递归展开相邻节点来定义（感觉以前讲过找不到了算了）

### How Expressive is a GNN?
- injectively 单射 different elemens into different outputs
- 如果GNN每一步骤的聚合都可以把他的邻居信息全部保存（injective neighbor aggregtation），这样生成的节点嵌入就可以相互区分，因此这样的GNN就是most expressive
![[Pasted image 20250108170813.png]]

### Summary so far 
![[Pasted image 20250108170951.png]]

## Designing the Most Powerful GNN 

### Expressive Power of GNN 
- expressive power可以通过其使用的邻居聚合函数的表达能力来表征
- 下一步就是理论分析aggregation functions
- 邻居聚合可以抽象为多集合（具有重复元素的集合）上的函数。
![[Pasted image 20250108172010.png]]

### Neighbor Aggregation  (Analyze aggregation function)
![[Pasted image 20250108172159.png]]

#### GCN mean-pool
-  GCN的聚合函数无法区分颜色比例相同的不同多集
- ![[Pasted image 20250108172309.png]]
- ![[Pasted image 20250108172327.png]]
- [[cs224w 21Fall 08 Applications of GNN#Issue of Global Pooling]]类似，区分详见上文

#### GraphSAGE max-pool
![[Pasted image 20250108173028.png]]

### Summary so far 
![[Pasted image 20250108173111.png]]

### Designing Most Expressive GNNs 
- designing injective neighbor aggregation function over multi-Sets
- 我们如何定义单射多集函数？- 我们使用神经网络

### Injective Multi-Set Function 
![[Pasted image 20250108173638.png]]
- 这个应该是基本框架
- 我们可以得到Intuition：我们可以将$f$定义为产生one-hot的代表color编码为向量。将这些编码求和可以保存输入多集的所有信息，然后使用非线性向量得到单射的特性
![[Pasted image 20250108174231.png]]

### How to model $\Phi$ and $f$ in $\Phi \left( \sum_{x \in S}f \left( x \right) \right)$ -通用近似定理
- 即我们可以使用MLP近似任意精度的连续函数
- 我们实践通常使用100到500层
![[Pasted image 20250108174659.png]]
![[Pasted image 20250108174713.png]]

### Most Expressive GNN  GIN 图同构网络
![[Pasted image 20250108174859.png]]
- 其聚合函数是单射的，没有区分失败的案例，是信息传递类GNN中表示能力最强的GNN

![[Pasted image 20250108175041.png]]
- 我们可以从另一个角度看待GIN即第二讲的WL graph kernel 
>[!tip] 
>GIN与WL graph kernel的关系:  
>- 我们通过将GIN与WL graph kernel（获得图级别特征的传统方法）做关联，来全面了解GIN模型。  
>- GIN可以说是WL graph kernel的神经网络版

### Relation to WL Graph Kernel 
- recall : [[cs224w 21Fall 02 Tradition-ml#Weisfeiler-Lehman KERNEL]]
![[Pasted image 20250108175254.png]]
- 具体color refinement 省略
![[Pasted image 20250108175356.png]]
WL graph kernel就像个硬编码的图神经网络。
- 算法：color refinement
- 迭代公式见上，迭代至稳定后，如果两个图的颜色集相同，说明他们同构

### The Complete GIN Model 

- GIN就用一个神经网络来建模这个单射的哈希HASH函数。
![[Pasted image 20250108175438.png]]
![[Pasted image 20250108175449.png]]
- 如果输入特征（即初始颜色）是独热编码，那么直接加总就是单射的（跟上面的例子一样），我们就仅需 $\Phi$ 来确保函数的单射，它需要产生独热编码来作为下一层的输入特征。
![[Pasted image 20250108180621.png]]
![[Pasted image 20250108180746.png]]
![[Pasted image 20250108180756.png]]

### Epressive Power of GIN 
![[Pasted image 20250108180825.png]]

### The Power of pooling
![[Pasted image 20250108181110.png]]

### Improving GNN's Power 
![[Pasted image 20250108181132.png]]
>[!tip] 
>增加GNN的表示能力  :
>- 对于类似“节点处于不同环中”这种问题，GNN仍然无法区分（因为计算图相同）。
>- 解决方法可以是添加可区分节点的feature，也可以使用reference node来区分相同计算图等。后续课程将会讲述具体做法。

## Summary 
![[Pasted image 20250108180925.png]]
