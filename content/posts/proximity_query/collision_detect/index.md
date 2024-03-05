+++
title = 'collision detect'
date = 2024-02-26T19:08:59+08:00
draft = true
+++

# collision Detect

# CCD的邻接三角形对Culling

## 算法流程

![avatar](pic/AdjacentTriangleCullingAlgorithm.png)

## 邻接三角形对类型

相较于刚性模型或者铰接模型，可形变模型的CCD的效率主要受到自碰撞的影响。  
自碰撞可进一步分为两种类型：相邻三角形之间的自碰撞和非相邻三角形之间的自碰撞
在CCD中，相邻三角形对计算占主要部分

AVTP  
AETP  
NTP

### 静态潜在碰撞对（static potential colliding feature pairs）

这些特征对是作为来自所有相邻三角形对（AVTPs和AETPs）的预处理的一部分而生成的。它们被收集一次，并在整个模拟过程中保持不变。之后的模拟步长中，将会忽略邻接三角形对而是直接从中进行culling

### 动态潜在碰撞对（dynamic potential colliding feature pairs）

若NTP对中出现包围盒碰撞，则将其加入，并在每个模拟步长中进行更新

## AVTP Culling

![avatar](pic/AVTP-Culling.png)

若满足该条件，所有的9个初等检验都可以被剔除。否则，根据该定理失败的情况，算法将$CCD_{sub}(t_d，t_b)$、$CCD_{sub}(t_a,t_c)$或两者对应的特征对记录为静态潜在碰撞特征对。

## AETP Culling

![avatar](pic/AETP-Culling.png)

若满足该条件可以跳过AETP中的所有4个测试，否则根据根据失败的原因，将$CCD_{sub}(t_a,t_c)$或$CCD_{sub}(t_b,t_d)$或者二者都作为静态潜在碰撞特征对

该定理的几何意义在于，除了AETP处于边界（即$t_c$或$t_d$不存在）外，几乎所有所有与AETP相关的特征对已经被AVTP测试阶段覆盖。因此，在大多数情况下，都可以跳过这些特征对。

## 基于表的重复消除

将特征对存储为$\left[\left\{e_{i}, e_{j}\right\}, r_{i j}\right]$或者$\left[\left\{v_{k}, t_{l}\right\}, r_{k l}\right]$。其中$r_{ij}$和$r_{kl}$分别是特征对${e_i、e_j}$和${v_k、t_l}$的基本测试结果。  

对于需要进行精确碰撞测试的特征对，我们首先在特征测试表中搜索这对特征对。如果已经测试了特性对，则将返回存储的结果。否则，将调用三次方程求解器来计算特征对之间的接触时间。然后将接触时间保存到特征测试表中，并结果返回。  

表搜索策略非常简单而有效。通过为每个特征（即边、顶点和三角形）分配一个唯一的id，特征测试表可以实现为一个哈希表。在基准测试中，哈希表实现在删除所有重复项方面是相当有效的。由于使用基于邻接的culling已经减少了很大一部分的假阳性，因此基于表的重复消除减少了许多其他特征对。

## 参考文献

[Tang etc - 2008 - Adjacency-based culling for continuous collision d.pdf](./paper)