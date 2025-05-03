---
tags:
  - GNN
  - Stanford
date: 2025-01-03
---
## Graph Neural Networks

### Recap: Node embedding [[cs224w 21Fall 03 Node Embedding]]
![[Pasted image 20250103153730.png]]

### Today: Deep Graph Encoders
![[Pasted image 20250103153459.png]]
![[Pasted image 20250103153543.png]]
- encoder 不再是简单的“shallow” Encoding (embedding-lookup)
- encoder 现在是多层非线性变换基于图结构

### Task we can solve
![[Pasted image 20250103153911.png]]
### Model ML Toolbox and ML for graph 图机器学习的独特之处
![[Pasted image 20250103154044.png]]
- 图更加复杂，图没有固定的顺序或者参考点
- 图经常是动态的有多种特征

## Basics of Deep Learning 简略

### ML as  Optimization 
![[Pasted image 20250103154241.png]]
![[Pasted image 20250103154326.png]]
### Loss Function Example 
![[Pasted image 20250103154414.png]]
>[!tip] 
>cross entropy 交叉熵好像经常用于多个类别的分类，对应 softmax 激活函数进行归一化，这个只是一个简略说明，后文还有回归的loss[[cs224w 21Fall 08 Applications of GNN#Classification and Regression Loss]]

### Gradient Descent 梯度下降 (Stochastic 随机)
![[Pasted image 20250103154633.png]]
![[Pasted image 20250103154640.png]]
- 每次计算所有数据集太复杂，我们随机挑选不同的minibatch小批量进行梯度下降
- 迭代次数等于整个数据集大小与批量的比例
- 可以对SDG进行一些优化
![[Pasted image 20250103154847.png]]

### Neural Network function
![[Pasted image 20250103155057.png]]
>[!tip] 雅各比矩阵与梯度。具体的再说
>![[Pasted image 20250103155506.png]]
>![[Pasted image 20250103155542.png]]

### Back-propagation 反向传播-计算梯度
- 使用链式法则反向传播梯度
![[Pasted image 20250103172830.png]]

### 非线性激活函数
![[Pasted image 20250103172903.png]]
### Muti-layer Perceptron MLP
![[Pasted image 20250103172932.png]]

### Summary 
![[Pasted image 20250103172950.png]]

## Deep Learning for Graphs 

### Overview
>[!abstract] 
>- Local network neighborhoods: 单个节点如何定义
>	- 描述聚合策略aggregation
>	- 定义计算图computation graphs
>- Stacking multiple layers: 将多个层堆叠
>	- 描述模型，参数，xunlian
>	- 如何拟合这个模型
>	- 监督学习和非监督学习简单例子

### Setup
![[Pasted image 20250103175953.png]]

### Graph Convolutional Networks 
![[Pasted image 20250103180305.png|500]]
![[Pasted image 20250103180451.png|500]]
![[Pasted image 20250103180646.png|500]]
- 我们定义一个计算图，节点从他邻居中聚合后转化嵌入

### Deep Model : Many Layers
![[Pasted image 20250103180830.png]]
- 模型可以有任意的深度
- 节点在每一层都有嵌入
- 第零层就是input特征
- k层嵌入获得k hops away 的节点信息

### Neighborhood Aggregation 聚合
不同神经网络的关键区别在于使用不同的方法在层之间聚合信息
![[Pasted image 20250103191006.png]]

### The Math: Deep Encoder  - basic Approach
- average neighbor messages and apply a neural network
![[Pasted image 20250103191201.png]]

### Training the model - 矩阵表示
![[Pasted image 20250103191933.png]]
- 需要训练的两个参数，$W_k$ 和 $B_k$ 分别对应邻居和自身
![[Pasted image 20250103192138.png]]
>[!tip] 
>$A_{v,:}$ 表示邻接矩阵中v的所有邻居，乘以embedding矩阵相当于求和

![[Pasted image 20250103193823.png]]
可以写成这个看起来很厉害的矩阵表达式，A是稀疏的，好像会方便计算什么的，反正就是很好

### Train GNN
![[Pasted image 20250104005002.png]]
有两种类型的任务一个监督一个非监督，非监督也可以看作自监督图结构作为监督条件

### Unsupervised and Supervised Training

![[Pasted image 20250104014717.png|500]]
>[!tip] 
>- DEC [[cs224w 21Fall 03 Node Embedding#Embedding nodes]]
>- 无监督学习关注图结构
>- 图相似性 [[cs224w 21Fall 03 Node Embedding#How to Define Node Similarity]]
>- 图中说的例子关键概念有点模糊，节点邻近性只是一个表示节点之间关系的概念，有多种表示形式，比如路径长度，余弦相似度等等。矩阵分解的loss只是另一种表示方法

![[Pasted image 20250104021009.png|500]]
![[Pasted image 20250104021032.png]]

### Model Design: Overview
>[!abstract] Overview
>1. 首先定义邻域聚合函数
>2. 定义嵌入时的损失函数
>3. 在一个节点集中训练即一批量计算图
>4. 生成节点嵌入

![[Pasted image 20250104021520.png]]
![[Pasted image 20250104021531.png]]
![[Pasted image 20250104021540.png]]

### Inductive Capability 归纳能力 (transductive与inductive辨析)
- transductive：无法将模型泛化到训练时没有出现的数据上，在训练时就要应用全部训练集和测试集，举例：shallow encoding[[cs224w 21Fall 03 Node Embedding#“Shallow" Encoding]]  
- inductive：在训练集上训练模型，模型可以用以预测在训练时没有出现过的测试集的数据（举例：本文中所讲的GNN，MLP等一般模型）
![[Pasted image 20250104022030.png]]
![[Pasted image 20250104022048.png]]
![[Pasted image 20250104022057.png]]

### Summary 
![[Pasted image 20250104022212.png]]

## Graph Convolutional Networks and GraphSAGE

### Overview
GraphSAGE:
- 这个聚合函数可以是任何将一组向量（节点邻居的信息）映射到一个向量上的可微函数： $h_{v}^{ \left( l+1 \right)}= \sigma \left( \left[ W_{l} \cdot AGG \left( \left \{ h_{u}^{ \left( l \right)}, \forall u \in \right. \right. \right. \right.$   $\left. N \left( v \right) \right \}),B_{l}h_{v}^{ \left( l \right)} \left. \right])$  
- 与上文主要区别就是泛化聚合所用函数
- 在每一层的节点嵌入上都可以做L2归一化： $h_{v}^{k}\leftarrow\frac{h_{v}^{k}}{|| h_{v}^{k} ||_{2}}$ （有时可以提升模型效果）

### GraphSAGE Idea 
![[Pasted image 20250104023045.png]]
![[Pasted image 20250104023158.png]]

### Neighborhood Aggregation 对比
![[Pasted image 20250104023324.png]]
![[Pasted image 20250104023331.png]]
- LSTM 是啥我不到
## summary
![[Pasted image 20250104023404.png]]
