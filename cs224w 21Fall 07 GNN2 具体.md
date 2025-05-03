---
tags:
  - GNN
  - Stanford
date: 2025-01-05
---
## A General Perspective on GNN 整体观

### Recap
[[cs224w 21Fall 06 GNN1 概述idea]]
![[Pasted image 20250105133631.png]]
- Idea: Raw input graph ≠ computational graph
	- Graph feature augmentation
	- Graph structure augmentation
- Supervised/Unsupervised objectives
- Node/Edge/Graph level objectives

## A Single Layer of a GNN 单层

### A Single GNN layer
![[Pasted image 20250105133958.png]]

### Message Computation 节点特征转化
![[Pasted image 20250105134233.png]]
>[!tip] 
>- 上一讲[[cs224w 21Fall 06 GNN1 概述idea#The Math Deep Encoder - basic Approach]]中没有强调massage computation部分，~~直接使用节点嵌入$h_u$并将其归一化~~。错啦，使用了矩阵进行变化，被上一讲的符号顺序迷惑了😭
>- 这里使用了线性变换处理了节点本身的特征或者说是嵌入

### Message aggregation
![[Pasted image 20250105135714.png]]
>[!tip] 
>- 上一讲的GraphSAGE和简单的GCN就差在这里[[cs224w 21Fall 06 GNN1 概述idea#Graph Convolutional Networks and GraphSAGE]]

### Message Aggregation : issues 没有自身特征
![[Pasted image 20250105140006.png]]
>[!tip] 
>- concatenation 串联连结应该就是一些激活函数
>- 激活函数也可以运用到AGG和MSG部分让他们变得不再线性
### Summary 
![[Pasted image 20250105140424.png]]

### Classical GNN layers: GCN
[[cs224w 21Fall 06 GNN1 概述idea#Graph Convolutional Networks]]
![[Pasted image 20250105140704.png]]
![[Pasted image 20250105140715.png]]
这个比上一讲的模型更加简略没有自身特征
### Classical GNN layers: GraphSAGE
![[Pasted image 20250105144650.png]]
这些基本重复了，参考上一讲[[cs224w 21Fall 06 GNN1 概述idea#GraphSAGE Idea]]
>[!tip] 
>嗯$W$这个线性变幻矩阵好像在外面也可以在里面

![[Pasted image 20250105145254.png]]
![[Pasted image 20250105145316.png]]

### Classical GNN Layers: GAT Graph Attention Networks 
![[Pasted image 20250105145738.png]]
>[!tip] 
>- 第五讲有相似之处，belief propagation[[cs224w 21Fall 05 Label Propagation for Node Classification#Loopy BP Algorithm and Notation 公式算法]]
>- ~~这些概念应该都是相关联的~~，一个是神经网络一个是简单的迭代得到节点重要性，感觉也不是一样的
>- GAT重点在于节点的邻居不在2同等重要，可以被学习
>	![[Pasted image 20250105150336.png]]

### Attention Mechanism 定义注意力矩阵
- 节点关注邻居的信息
- 隐式地为邻居中的不同节点指定不同的权重
- 让$\alpha_vu$作为一个注意力机制$a$的副产品
- 根据节点的message得到注意力系数$e_vu$再使用softmax处理得到最后$\alpha_vu$

![[Pasted image 20250105150908.png]]
![[Pasted image 20250105151010.png]]
![[Pasted image 20250105151241.png]]
- 注意力机制中的系数可以和权重矩阵（NN中的系数）同时训练
![[Pasted image 20250105151901.png]]
>[!tip] 
>- multi-head attention：稳定注意机制的学习过程
>- 使用不同注意力分数（即使用不同的系数）训练多个注意力系数然后进行聚合得到更稳定的注意力系数，感觉是使用不同的初始值得到不同结果？不知道了

### Benefits of Attention Mechanism / Summary 
![[Pasted image 20250105152630.png]]
 核心优点：隐式定义节点信息对邻居的importance  $\alpha_{vu}$

### GNN Layer in Practice
![[Pasted image 20250105153052.png]]
- 在实际使用中，上文的GNN通常看作一个起始点，我们考虑一个通用GNN层
- 我们使用现代deep learning模型转化gnn得到的嵌入
![[Pasted image 20250105153441.png]]

### Batch normalization 数据处理
![[Pasted image 20250105153614.png]]
将得到的嵌入归一化：
- 均值变成0
- 饭菜变成单位方差
应该可以让下游任务计算更简单

### Dropout 防止过拟合
![[Pasted image 20250105153858.png]]
训练时随机忽略一些神经网络中的节点，在message时的线性层
![[Pasted image 20250105153949.png]]

### Activation 非线性激活
![[Pasted image 20250105154144.png]]

### Summary 
![[Pasted image 20250105154211.png]]

## Stacking Layers of a GNN 

### Stacking GNN Layers 
![[Pasted image 20250105154441.png]]
两种堆叠链接GNN层的方法：
- 按顺序地堆叠层
- 添加一些跳过链接的方法
![[Pasted image 20250105154812.png]]

### The Over-smoothing problem 过度平滑
按顺序地堆叠多个GNN层会带来过度平滑的问题，即所有节点聚合到相同的嵌入值，原因与下文感受野的概念相关
![[Pasted image 20250105155039.png]]

### Receptive Field of a GNN & Over-smoothing 接受野
![[Pasted image 20250105155244.png]]
![[Pasted image 20250105155306.png]]
>[!tip] 
>接受野就是节点嵌入时接受到的所有其他节点范围的抽象概念
>k-hop就是GNN的层数影响接受野
>随着层数的增加，节点间的接受野重叠越来越多导致嵌入趋近于相同

![[Pasted image 20250105155721.png]]

### Design GNN Layer Connectivity 解决过度平滑问题
![[Pasted image 20250105155954.png]]
>[!tip] 
>- 为了避免过度平滑，我们不能叠加太多层，与此同时我们需要加强神经网络的效果。
>- 我们有两个方法：
>	- 增加单层GNN的表现：在每层GNN增加表现力；添加不传递message的层
>	- 加入跳过链接

### Expressive Power for Shallow GNNs 01
![[Pasted image 20250105160350.png]]
>[!tip] 
>- 上文单层神经网络中的[[cs224w 21Fall 07 GNN2 具体#Summary|- 激活函数也可以运用到AGG和MSG部分让他们变得不再线性]]只是将线性变成非线性，这种方法更力
>- 上文在单层神经网络之后进行transformation还是堆叠常规的神经网络层与这个不同[[cs224w 21Fall 07 GNN2 具体#GNN Layer in Practice]] 02 与这个很像
>- 01和02方法还不一样，01还保留了GNN的框架。嗯概念差距细微

### Expressive Power for Shallow GNNs 02
![[Pasted image 20250105162039.png]]
>[!tip] 
>- pre-processing layers：如果节点特征必须经过编码就很重要（如节点表示图像/文字时）
>- post-processing layers：如果在节点嵌入的基础上需要进行推理和转换就很重要（如图分类、知识图谱等任务中）colab2中的图分类任务就都有，用AtomEncoder作为pre-processing layer，池化层作为post-processing layer(待补)

### Skip connections in GNNs
![[Pasted image 20250105162417.png]]
我们可以通过在GNN中添加快捷方式来增加早期层对最终节点嵌入的影响，靠前的GNN层可能能更好地区分节点
![[Pasted image 20250105162741.png]]
![[Pasted image 20250105162814.png]]
>[!tip] 
>注意区分聚合节点自身特征的公式[[cs224w 21Fall 07 GNN2 具体#Message Aggregation issues 没有自身特征]]

### Other Options of Skip Connections
![[Pasted image 20250105163052.png]]

## Graph Manipulation in GNNS 
这个后面好像会讲，视频没讲，累死了不正了到时候补个链接
孩子，他就在下一讲😭[[cs224w 21Fall 08 Applications of GNN]]

## Summary 
![[Pasted image 20250105163156.png]]
