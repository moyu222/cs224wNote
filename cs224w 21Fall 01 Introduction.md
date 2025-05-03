---
tags:
  - GNN
  - Stanford
date: 2025-01-02
---
## Why Graphs?

- Graphs are a general language for describing and analyzing entities with relations/interactions
- 图是研究带有关系和互动的实体的普适语言，可显性地对关系建模，得到更好的表现
- arbitrary size and complex topological Structures
- no fixed node ordering or reference point
- often dynamic and have multimodal features
![[Pasted image 20241230202859.png]]
### 传统的有监督机器学习过程
![[Pasted image 20241230203628.png]]
在传统机器学习流程中，我们需要对原始数据进行特征工程feature engineering (手动提取特征)，但是现在我们使用表示学习representation learning的方式自动提取数据特征，直接应用于下游预测任务

### 图的表示学习：Map nodes to d-dimensional embeddings 

- __similar nodes in the network are embedded close to together__
![[Pasted image 20241230204207.png]]

### Course Outline 大纲

We are going to cover various topics in Machine Learning and Representation Learning for graph structured data:

▪ Traditional methods: Graphlets, Graph Kernels
▪ Methods for node embeddings: DeepWalk, Node2Vec
▪ Graph Neural Networks: GCN, GraphSAGE, GAT, Theory of GNNs
▪ Knowledge graphs and reasoning: TransE, BetaE 
▪ Deep generative models for graphs: GraphRNN 
▪ Applications to Biomedicine, Science, Industry

## Applications of Graph ML

 4种不同类型的任务：
![[Pasted image 20241230204501.png]]
### Node level: Node classification - predict property of a node

### Edge level : Link prediction 

推荐系统	
	![[Pasted image 20241230205139.png]]
PinSage基于图的推荐系统[^1]	
	![[Pasted image 20241230205635.png]]
### Graph level: Categorize different graphs
graph generation: Drug discovery
graph evolution: Physical simulation
### Community (subgraph) level: Detect if nodes form a community

## Choice of a graph representation:

▪ Directed, undirected, bipartite, weighted, adjacency matrix



# Footnotes

[^1]: Ying et al., Graph Convolutional Neural Networks for Web-Scale Recommender Systems, KDD 2018