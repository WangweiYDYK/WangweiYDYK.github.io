+++
title = 'Point to Mesh Distance P2M'
date = 2024-02-26T18:26:30+08:00
draft = true
+++

# P2MSolver

## basic defination
![Alt text](<93GHS3J){OC]ODBRP2((5V1.png>)
图a，b分别代表$e$和$f$的vertical space，记作$Space^{\perp }(e)$和$Space^{\perp }(f)$ 

![Alt text](<@67(`QA{P2~ZKAIIVF4H9FG.png>)
图a为point $v$ 和segment $l_e$ 的bisector surface，图b为point和plane的bisector surface。bisector surface将一个whole space划分成convex part和non-convex part，其中point处于convex part,在这里用$Bisect^v(v,l_e)$代表包含$v$的convex part,$Bisect^e(v,l_e)$代表non-convex part

## main idea
二维情况下
$\operatorname{Cell}\left(v ; \mathcal{V}_V\right) \cap \operatorname{Cell}\left(e ; \mathcal{V}_{E}\right) \neq \emptyset$，则称$v$ interception $e$
其中$\mathcal{V}_V$代表顶点V的Voronoi图，$\mathcal{V}_E$代表边E的Voronoi图
![Alt text](<p2m_example.png>)
如图所示，对于任意一点点q，如果到$v_1$的距离近，则距离最近的线段可能为$e_1, e_2, e_3$，如果到$v_2$的距离比较近，则最近的线段可能为$e_1, e_3$。

对于三维情况，如下图所示

