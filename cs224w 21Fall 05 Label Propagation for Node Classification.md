---
tags:
  - GNN
  - Stanford
date: 2025-01-03
---
## Message Passing and Node Classification 信息传递和节点分类

### Outline
![[Pasted image 20250103131645.png]]
![[Pasted image 20250103131816.png]]
>[!abstract] Outline
>- 主要问题：给定一个一些节点带有标签的网络，我们怎么分配节点给网络中的其他节点
>- 我们讨论一个可选的框架：消息传递message Passing
>- Intuition: 相关性存在于网络中。
>	- 相似的节点连接
>	- key concept is collective classification协同分类
>- techniques:
>	- Relational classification
>	- Iterative classification
>	- Belief propagation

### Why correlations exist in network 相关性的原因
Homophily 和 Influence 同质性和影响
- 有着相似特点的节点具有同质性，更容易链接
- 有着链接的节点对会相互影响，使得节点特征即标签相似
![[Pasted image 20250103132704.png]]

### How to leverage this correlation 如何量化
Motivation1: 相似的节点互相连接
- ![[Pasted image 20250103132842.png]]
Motivation2: 节点的标签取决于：
- 节点自身的特征
- 他邻居节点的标签
- 他邻居节点的特征

### Semi_supervised Learning  半监督
![[Pasted image 20250103133136.png]]
![[Pasted image 20250103133155.png]]

### Approach : Collective Classification - 协作分类Overview 
![[Pasted image 20250103133313.png]]
>[!abstract] Overview
>- Intuition: 使用相关性对相互连接的节点进行同时分类
>- 概率框架
>- 使用马尔可夫假设：节点的标签取决于其邻居的标签
>	- $P(Y_v) = P(Y_v|N_v)$
>- 协同分类有三步：
>	- ![[Pasted image 20250103133743.png]]
>-

Local Classifier：用于初始化标签分配：
- 基于节点自身的特征attributes/features进行预测
- 标准分类任务
- 没有使用网络信息
Relational Classifier：捕捉相关性 Capture correlations
- 学习一个分类器基于邻居节点的标签或者特征进行预测标签
- 使用网络信息
Collective Inference：传播相关性 Propagate the correlation
- 对每个节点迭代使用关系分类器
- 迭代直到相邻标签之间的不一致性最小化
- 网络结构影响最终预测

### Problem Setting
![[Pasted image 20250103134522.png]]
![[Pasted image 20250103134823.png]]

## Relational Classification and Iterative classification

### Probailistic Relational Classifier 概率关系分类器
![[Pasted image 20250103135009.png]]
- 对未分类的初始化为0.5
- 更新直到收敛或最大迭代数
![[Pasted image 20250103135214.png]]
- Challenges:
	- 不一定收敛
	- 没有使用节点信息

### example
![[Pasted image 20250103135437.png]]
![[Pasted image 20250103135446.png]]
![[Pasted image 20250103135512.png]]

### Iterative Classification 迭代分类器
- 关系分类器没有使用节点特征，我们如何量化他们
- Main Idea：使用节点自身特征和邻域一起分类
![[Pasted image 20250103135716.png]]

### Compute the summary $Z_v$ 节点邻域信息
![[Pasted image 20250103135836.png]]
3种不同做法：
- 邻域不同标签数量或分布
- 出现最多次的节点
- 不同标签的数量

### Architecture of Iterative Classifiers 架构
![[Pasted image 20250103140132.png]]
>[!tip] 
>- 好像跟上一讲的消息传递类似[[cs224w 21Fall 04 Link Analysis PageRank(graph as matrix)]]，上一讲评估节点的重要性是连续值，这个应该是离散值。
>- 总感觉有一股即视感，应该后面有类似的吧，嗯就在下一讲gnn[[cs224w 21Fall 06 GNN1 概述idea#The Math Deep Encoder - basic Approach]]
>- 区别在于gnn在聚合之后使用激活函数，侧重点不同。这一讲主要是分类问题不需要嵌入到低维

### Example: Web Page classification
![[Pasted image 20250103141417.png]]
![[Pasted image 20250103141508.png]]
- $f_v$ 是节点自身的特征向量，例如网页上的不同词汇
- 简单的分类出现错误，我们使用其邻域以及自身特征迭代
![[Pasted image 20250103141929.png]]
### Iterative Classifier 训练过程
![[Pasted image 20250103142211.png]]
- 训练集上训练
![[Pasted image 20250103142236.png]]
![[Pasted image 20250103142255.png]]
![[Pasted image 20250103142304.png]]
![[Pasted image 20250103142322.png]]
![[Pasted image 20250103142612.png]]
![[Pasted image 20250103142620.png]]
### summary
>[!summary] 两个方法进行协同分类
>关系分类：迭代更新节点属于标签的概率根据他的邻居节点
>迭代分类：将节点自身特征纳入考虑

## Collective Classification: Belief Propagation - 信念传播本节协调分类最后一个方法

### Loopy Belief Propagation - 循环信念传播
![[Pasted image 20250103143233.png]]
>[!tip] 
>应该就是信息传播，更上一讲差不多，只是这次传播的是节点属于某类的概率

### Message Passing 
![[Pasted image 20250103143634.png]]
>[!tip] 
>这个数节点数量应该只是一个例子

### Generalizing to a Tree
![[Pasted image 20250103143744.png]]

### Loopy BP Algorithm  and Notation 公式算法
![[Pasted image 20250103143823.png]]
![[Pasted image 20250103144013.png]]
>[!tip] 
>- 第一个那个标签矩阵记得于后文一个叫self-attention（好像叫这个）有联系，链接待补。找到啦[[cs224w 21Fall 07 GNN2 具体#Classical GNN Layers GAT Graph Attention Networks]]
>- 简单来说就是在当前节点邻居a属于类0的情况下，当前节点属于类1的概率，这是个标签数量成标签数量的方阵
>- Prior belief 就是要传递的belif一个概率

![[Pasted image 20250103144504.png]]
嗯很复杂的样子
![[Pasted image 20250103144801.png]]
>[!tip] 好像是另一个关键概念
>以上我们考虑的情况是，节点有着一定的顺序，path时候就是从一头到另一头。至于泛化到树状结构就是从leaf传递到root这样。上文应该有一页ppt询问如果有循环就是cycle怎么办，我忘截图了，下文要讲。其实真实图没有很多循环

### What if has cycles?
![[Pasted image 20250103145339.png]]![[Pasted image 20250103145404.png]]

### Advantages of Belief Propagation 
![[Pasted image 20250103150828.png]]

## Summary 
![[Pasted image 20250103150844.png]]
