---
tags:
  - GNN
  - Stanford
date: 2025-01-05
---
## Graph Augmentation for GNNS 图增广

### Why Augment Graph 
![[Pasted image 20250106130515.png]]
>[!tip] 
>- 输入图缺少特征，这里所说的应该是整个图结构上的结构特征。嗯也可以创造节点特征（感觉节点重要性就是第四讲的[[cs224w 21Fall 04 Link Analysis PageRank(graph as matrix)]]可以作为特征）
>- 图有时太稀疏有时太密集有时太大

### Graph Augmentation approaches
根据上文的缺陷进行改进
![[Pasted image 20250106130740.png]]
>[!tip] 
>最后一个采样子图计算嵌入后文待表
>- 缺少特征进行特征增广
>- 太稀疏增加虚拟节点或链接
>- 太密集在进行消息传递时采样邻居

### Feature Augmentation on Graphs 缺少特征

#### 节点自身没有特征
![[Pasted image 20250106131521.png]]
- 简单分配常数值给节点
![[Pasted image 20250106131406.png]]
- 为每个节点分配节点id创造one-hot向量的节点特征
![[Pasted image 20250106131612.png]]
- one-hot 方法虽然很多方面表现优秀但是inductive能力弱，而且大多应用在小图中

#### 某些结构GNN很难学习
![[Pasted image 20250106131907.png]]
![[Pasted image 20250106131922.png]]
>[!tip] 
>- 不能分辨某些结构下的节点，因为他们的计算图相同
>- 后文待补，记得后的了[[cs224w 21Fall 09 Theory of GNN#Computational Graph 计算图]]。概念错综复杂啊

![[Pasted image 20250106132117.png]]
看起来好简单粗暴的方法，和上文的方法应该类似
![[Pasted image 20250106132229.png]]
>[!tip] 
>是这样的了，上文说的利用PageRank得到了验证

### Add Virtual Nodes / edges 图太稀疏
![[Pasted image 20250106132358.png]]
>[!tip] 
>很常见的方法是增加2-hop的虚拟链接，这也很有现实意义。例如在作者，论文的二分图中表示author-author合作的关系，嗯具体来说就是$A + A^2$

![[Pasted image 20250106132637.png]]
>[!tip] 
>这个增加虚拟节点的理解是，因为GNN的层数应该不能很大，导致有些节点可能在未来会有链接（这里说的有些具体了，就是有些距离虽远的节点还是有影响对方）。这时我们可以添加一个虚拟节点，例如让所有节点又有距离为2，这样就能显著提高信息传递的能力

### Node Neighborhood Sampling 图太密集-论文补充
![[Pasted image 20250106133134.png]]
![[Pasted image 20250106133301.png]]
![[Pasted image 20250106133318.png]]
![[Pasted image 20250106133328.png]]
>[!tip] 
>- 在复杂图中计算每个层节点嵌入可能太复杂，我们在不同层中随机选择一些邻居进行节点嵌入，可以显著降低计算成本而且表现良好
>- 上文说过的scalling 待补

## Prediction with GNNS 

### A General GNN Framework 
![[Pasted image 20250106133829.png]]
我们定义完GNN的框架后，使用具体的学习目标进行训练

### GNN Training Pipeline - prediction head
![[Pasted image 20250106134019.png]]
本章主要就是讲预测问题，因此是不同的prediction head ：
- Node-level Tasks
- Edge-level Tasks
- Graph-level Tasks 
![[Pasted image 20250106134318.png]]

### Prediction Heads: Node-level
![[Pasted image 20250106134428.png]]

### Prediction Heads: Edge-level 
![[Pasted image 20250106134511.png]]
edge-level 的问题需要定义edge head有两个方法：
- concatenation + linear 这个方法可以定义k个不同的链接 k-way
- dot product ~~这个只能预测一个链接的存在~~，也可以使用multi-head进行k-way 
![[Pasted image 20250106134829.png]]
![[Pasted image 20250106134948.png]]

### Prediction Heads: Graph-level 
![[Pasted image 20250106135600.png]]
>[!tip] 
>将所有节点嵌入进行转化类似GNN中的AGG( )方法

![[Pasted image 20250106135732.png]]

### Issue of Global Pooling
![[Pasted image 20250106135947.png]]
>[!tip] 
>- 有些pooling方法会丢失信息，我们可以使用分层聚合的方法，第三讲曾经提过[[cs224w 21Fall 03 Node Embedding#Preveiw Hierarchical Embedding 分层嵌入]]
>- 后文[[cs224w 21Fall 09 Theory of GNN#GCN mean-pool]]很类似，这个是在应用层面graph-level时sum pooling 时不能区分，使用分层聚合解决。后文那个是单个层中分析expressive power中的聚合函数能力

### Hierarchical Global Pooling 分层嵌入
![[Pasted image 20250106140848.png]]
>[!tip] 
>- 将整个节点分成不同部分进行分层嵌入，下图很清晰。
>- 感觉可以和后文的聚类进行结合待补，嗯只是两个不同GNN可以并行进行

### Hierarchical Pooling In Practice 
![[Pasted image 20250106141356.png]]
![[Pasted image 20250106141529.png]]
训练GNN  B在后文😊待补

## Training Graph Neural Networks 

### GNN Trainng Pipeline 2 - predictions and label 
![[Pasted image 20250106141826.png]]

### Supervised vs Unsupervised 
![[Pasted image 20250106141931.png]]
讲过好多遍不说了

### Supervised and Unsupervised Labels on Graphs 
![[Pasted image 20250106142047.png]]
![[Pasted image 20250106142226.png]]
这些问题都讲了或者以后会讲，嗯

### GNN Training Pipeline 3 and Setting  - loss function
![[Pasted image 20250106142810.png]]
![[Pasted image 20250106142941.png]]

### Classification and Regression Loss 
![[Pasted image 20250106143044.png]]
>[!tip] 
>[[cs224w 21Fall 06 GNN1 概述idea#Loss Function Example]]第六讲简单讲过
#### Classification Loss 
![[Pasted image 20250106143130.png]]
#### Regression Loss 
![[Pasted image 20250106143445.png]]

### GNN Training Pipeline 4 - evalution metrics 评价指标
![[Pasted image 20250106143729.png]]
#### Evalution Metrics: Regression 
![[Pasted image 20250106143809.png]]
>[!tip] 
>回归得到的值是连续的所有应该用RMSE 和 MAE 这种连续的error继续评估
#### Evalution Metrics: Classification 
![[Pasted image 20250106143918.png]]

### Metrics for Binary Classification 二分类
![[Pasted image 20250106144434.png]]
![[Pasted image 20250106144509.png]]
![[Pasted image 20250106144759.png]]
ROC AUC  
ROC曲线下面积。越高越好，0.5是随机分类器，1是完美分类器。  
随机抽取一个正样本和一个负样本，正样本被识别为正样本的概率比负样本被识别为正样本的概率高的概率。

## Setting-up GNN Prediction Tasks 

### GNN Training Pipeline 5 - split our dataset 
![[Pasted image 20250106145518.png]]
![[Pasted image 20250106145704.png]]
有两个分割的方法固定和随机，随机比较好

### Why Splitting Graphs is Special 
![[Pasted image 20250106145953.png]]
>[!tip] 
>graph不像普通的神经网络，节点之间有着联系不能随便分割

### Transductive / Inductive Settings 
#### Solution 1 Transductive Setting 
![[Pasted image 20250106150212.png]]
>[!tip] 
>- 我们使用整个图进行训练测试和验证
>- 我们训练时我们使用整个图嵌入，但是只使用一些节点进行训练
>- 在其他步骤中使用其他相互独立的节点

#### Solution 2 Inductive Setting 
![[Pasted image 20250106150736.png]]
#### Summary 
![[Pasted image 20250106151013.png]]
>[!tip] 
>- inductive 方法显然更有泛化能力，使用不同图组啊不同步骤也符合这种直觉
>- 可以从另一个角度来看，随着时间的推进一些节点添加，一些链接增加

### Example: Graph Classification and Link Prediction 
![[Pasted image 20250106151402.png]]
在图的维度上我们只能使用inductive Setting

![[Pasted image 20250106151530.png]]
这个讲的是单个图，下面是图的数据集
### Setting up Link Prediction 
![[Pasted image 20250106151956.png]]
>[!tip] 
>- 预测边任务复杂一些，在每个集合中我们还要将他们分成两个部分，具体就是message edges 和 supervision edges 

#### Inductive Link prediction split 
划分出3个不同的图组成的split，每个split里的边按照第一步分成message edges和supervision edges
![[Pasted image 20250106152228.png]]
![[Pasted image 20250106152633.png]]

#### Option 2 Transductive Link prediction split 
![[Pasted image 20250106152743.png]]
- transductive 是链接预测的默认方法
>[!tip] 
>- 在一张图中进行切分：在训练时要留出验证集/测试集的边，而且注意边既是图结构又是标签，所以还要留出supervision edges（要不然还搞啥呢……）
>- 具体来说：- 
>	- 训练：用 training message edges 预测 training supervision edges
>	- 验证：用 training message edges 和 training supervision edges 预测 validation edges
>	- 测试：用 training message edges 和 training supervision edges 和 validation edges 预测 test edges
>- 链接越来越多，图变得越来越稠密的过程。这是因为在训练过程之后，supervision edges就被GNN获知了，所以在验证时就要应用 supervision edges 来进行 message passing（测试过程逻辑类似）

![[Pasted image 20250106153205.png]]
![[Pasted image 20250106153330.png]]
![[Pasted image 20250106153355.png]]
![[Pasted image 20250106153404.png]]![[Pasted image 20250106153404 1.png]]

## Summary 
![[Pasted image 20250106153725.png]]
