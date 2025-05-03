---
tags:
  - GNN
  - Stanford
date: 2025-01-02
---
## Graph as matrix, Overall
- 这一讲我们通过矩阵视角进行图分析
- Treating a graph as a matrix allows us to:
	- 通过随机游走(PageRank)确定节点重要性
	- 通过矩阵分解得到节点嵌入
	- 使用矩阵分解看待其他节点嵌入
- 随机游走，矩阵分解和节点嵌入紧密联系

## PageRank (aka the Google Algorithm)

### Example: The web as a graph
![[Pasted image 20250102184113.png]]
![[Pasted image 20250102184132.png]]
- Web as a 有向图 directed graph
- web page are not equally "important"
- ==So, let's rank the pages using the web graph link structure==
- 我们使用以下link analysis approachs 去计算图中节点的重要性
	- PageRank
	- Personalized PageRank PPR
	- Random Walk with Restarts

### PageRank: The "Flow" model
- 我们将入联链接事做投票，计算一个节点的入联链接
- 一个节点重要如果被其他重要节点指入
- 递归问题
![[Pasted image 20250102184825.png]]
![[Pasted image 20250102184943.png]]
>[!tip] 
>使用高斯消元可以解决，但是不好
>我们使用矩阵

### PageRank: Matrix Formulation
![[Pasted image 20250102185059.png]]
![[Pasted image 20250102185420.png]]

### Connection to Random walk
![[Pasted image 20250102185519.png]]

>[!tip] 
>- 使用随机游走思想看待PageRank，从每个步骤角度直观看出。下文可以从另一个角度看，即整体游走多步看为矩阵的幂
>- p(t)为在t时所有节点的访问概率
>- 最后可能达到稳定状态，异常且看下文

![[Pasted image 20250102185820.png]]

### Recall Eigenvector of A matrix
![[Pasted image 20250102185931.png]]
![[Pasted image 20250102193131.png]]
>[!tip] 
>- 与第二讲特征向量中心性比
>- 由此看出上文式子看作特征值为1
>- 因为每一列和为1，这是个马尔可夫矩阵，1就是主特征值
>- 马尔可夫矩阵啥的我忘了，得复习线代了
>- 我们使用幂迭代解决r

>[!summary] PageRank: summary
>![[Pasted image 20250102193351.png]]

## PageRank: How to solve?

![[Pasted image 20250102193916.png]]

### Power Iteration Method
![[Pasted image 20250102194113.png]]
>[!tip] 
>![[Pasted image 20250102194332.png]]

![[Pasted image 20250102194401.png]]

### PageRank: Three Questions
>[!question] 
>- Does this converge?
>- Does it converge to what we want?
>- Are results reasonable?

### PageRank: Problems
![[Pasted image 20250102195540.png]]
- 死胡同不能收敛到我们想要的结果
- 蜘蛛陷阱即自环不能收敛
![[Pasted image 20250102195814.png]]
![[Pasted image 20250102195824.png]]

### Solution to Sipder Traps and Dead Ends
在每一步，随机访问者有两个选择：
- $\beta$ 概率随机访问一个链接
- $1-\beta$ 概率跳到一个随机节点
![[Pasted image 20250103004327.png]]

Teleports随机传送：再遇到死胡同是以概率为1.0进行随机传送
- 即随机访问其他节点
![[Pasted image 20250103004607.png]]
Why teleports solve problem? 应该不是很重要
![[Pasted image 20250103004653.png]]

### Google Solution: Random Teleports
![[Pasted image 20250103004828.png]]
![[Pasted image 20250103004937.png]]

### Example 
![[Pasted image 20250103005003.png]]
![[Pasted image 20250103005014.png]]
>[!summary] Solving PageRank: summary
>![[Pasted image 20250103005058.png]]

## Random Walk with Restarts and Persoalized PageRank

### Example: Recommendation 推荐系统
![[Pasted image 20250103005318.png]]
![[Pasted image 20250103005553.png]]
![[Pasted image 20250103005611.png]]
- 直觉：如果项目Q和P是由相似的用户交互，当用户与Q交互时，推荐P
- ==如何定义Proximity相似性==（图结构上的相似性邻近性）
### Node proximity Measurements
![[Pasted image 20250103010351.png]]
我们通过上面的图探讨如何定义相似性，最短路径和共同邻居都不合适（应该），反正使用Random walk with Restarts/Personalized PR

### Proximity on Graphs （概念一辨）
![[Pasted image 20250103011015.png]]
>[!tip] 
>- Personalized PR与PR只有一个区别：在随机传送时，我们不在随机传送到任意节点。我们只传送到节点S的子集，例如传送到节点S感兴趣的其他节点
>- Restart就是传送回起始节点

>[!tip] 邻近性与相似性
>- Proximity 邻近性与similarity 相似性：
>	- 邻近性关通常指节点在图中的距离或位置关系。它==强调的是节点之间的连接==
>	- 在邻近性较高的节点之间，通常存在边或较短的路径。例如，在社交网络中，两个用户如果通过少量共同好友连接，说明它们的邻近性较高。
>	- 相似性==可以关注结构特征也可以关注节点本身的特征==（后文记得讲过使用自身特征）[[cs224w 21Fall 05 Label Propagation for Node Classification#Architecture of Iterative Classifiers 架构]]
>	- 相似性可以通过计算节点属性的相似度（如余弦相似度、欧氏距离等）来得到。==即使两个节点在图中并不直接相连，如果它们的特征非常相似，依然可以认为它们是相似的==
>	- 邻近性关注的是图的结构性连接，而相似性则关注节点的属性或特征相似度
>	- 补充：邻近性只是概念以上都是表示方式详见[[cs224w 21Fall 06 GNN1 概述idea#Unsupervised and Supervised Training]]

>[!tip] 节点嵌入与PageRank 
>- **相似性**：PageRank可以被视为一种节点嵌入技术，特别是在图的结构信息方面。它提供了一种基于图结构的节点重要性评分。
>- **特征提取**：PageRank分数可以作为节点嵌入的特征之一，增强嵌入模型的表现，帮助捕捉节点的全局结构信息。
>- **算法结合**：有些节点嵌入方法可能会利用PageRank作为一种预处理步骤，或者在训练过程中结合PageRank的思想来改进嵌入质量。

>[!summary] Page Rank Variants
>![[Pasted image 20250103013412.png]]


### Idea: Random walks Algorithm
![[Pasted image 20250103012858.png]]
![[Pasted image 20250103013149.png]]
![[Pasted image 20250103013222.png]]

### WHy is this a good solution? 
![[Pasted image 20250103013315.png]]

### Summary
![[Pasted image 20250103013457.png]]

## Matrix Factorization and Node Embeddings

### Recall
![[Pasted image 20250103013738.png]]

### Connnection to Matrix Factorization
![[Pasted image 20250103013846.png]]
- 使用最简单的相似性定义，如果链接就相似
![[Pasted image 20250103014017.png]]
- 有对应的正确嵌入，所以不需要最大似然近似
- ==由边连通性定义节点相似度的内积解码器等价于𝑨的矩阵分解== - 使用矩阵分解而不是优化问题解决嵌入

### Random Walk-based Similarity 使用矩阵分解解决嵌入
![[Pasted image 20250103014628.png]]
![[Pasted image 20250103014549.png]]

### Limitations
![[Pasted image 20250103014718.png]]
![[Pasted image 20250103014727.png]]
![[Pasted image 20250103014735.png]]

## Summary
![[Pasted image 20250103014814.png]]