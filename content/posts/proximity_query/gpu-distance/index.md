+++
title = 'distance computation'
date = 2024-01-26T20:51:27+08:00
draft = false
tags = ['GPU']
categories = ["GPU", "Algorithm"]
+++

# Distance Computation

## CPU上的离散模型距离计算库
FCL
```cpp
#include <fcl/fcl.h>
typedef fcl::BVHModel<fcl::OBBRSS<REAL>> FCLModel;
std::vector<fcl::Vector3<REAL>> fclVerticesA, fclVerticesB;
std::vector<fcl::Triangle> fclTrianglesA, fclTrianglesB;
std::shared_ptr<FCLModel> geomA = std::make_shared<FCLModel>();
std::shared_ptr<FCLModel> geomB = std::make_shared<FCLModel>();
FCLModel* FCLModelA, *FCLModelB;

fcl::CollisionObject<REAL>* objA = new fcl::CollisionObject<REAL>(geomA, transformA);
fcl::CollisionObject<REAL>* objB = new fcl::CollisionObject<REAL>(geomB, transformB);
fcl::DistanceRequest<REAL> request;
request.enable_nearest_points = true;

fcl::DistanceResult<REAL> result;
fcl::distance(objA, objB, request, result);
```

PQP
```cpp
#include "PQP.h"
PQP_Model* modelA, * modelB;
PQP_REAL RA[3][3] = { rotA._data[0], rotA._data[3],rotA._data[6],rotA._data[1],rotA._data[4] ,rotA._data[7] ,rotA._data[2] ,rotA._data[5] ,rotA._data[8] };
PQP_REAL OFFA[3] = { offsetA.x, offsetA.y, offsetA.z };
PQP_REAL RB[3][3] = { rotB._data[0], rotB._data[3],rotB._data[6],rotB._data[1],rotB._data[4] ,rotB._data[7] ,rotB._data[2] ,rotB._data[5] ,rotB._data[8] };
PQP_REAL OFFB[3] = { offsetB.x, offsetB.y, offsetB.z };
PQP_DistanceResult dres;
PQP_Distance(&dres, RA, OFFA, modelA, RB, OFFB, modelB, 0.0, 0.0);
```

SSE SSE只提供了三角形计算的代码，且其中有部分有问题？目前修改了一版基于embree的可用的SSE代码

## 贝塞尔曲线的最小距离计算
参考Sum-of-square ccd的方法，计算贝塞尔曲线的距离
以下为matlab代码
```matlab
% u1是第一条贝塞尔曲线的参数，v1是第二条贝塞尔曲线的参数
sdpvar u1 v1 lambda;

% 默认两条都是三次贝塞尔曲线
d = 3;

% 分别为Bcurve1和Bcurve2的控制点
pta0 = [0;0;0];
pta1 = [8;0;0];
pta2 = [8;8;0];
pta3 = [0;8;0];

ptb0 = [-8;0;0];
ptb1 = [0;0;0];
ptb2 = [0;8;0];
ptb3 = [-8;8;0];

Bcurve1 = (1-u1)^3*pta0 + 3*(1-u1)^2*u1*pta1 + 3*(1-u1)*u1^2*pta2 + u1^3*pta3;
Bcurve2 = (1-v1)^3*ptb0 + 3*(1-v1)^2*v1*ptb1 + 3*(1-v1)*v1^2*ptb2 + v1^3*ptb3;

% sos约束数量取决于不等式约束的数量
gi = [u1*(1-u1); v1*(1-v1)];
[s1, s1c] = polynomial(u1, d);
[s2, s2c] = polynomial(u1, d);

% 待优化函数f 此处为距离函数
f = (Bcurve2 - Bcurve1)' * (Bcurve2 - Bcurve1);

C1 = [sos(s1); sos(s2)];
C2 = sos(f-lambda-[s1, s2] * gi);

F = [C1, C2];
[C, obj] = sosmodel(F, -lambda, [], [s1c; s2c; lambda]);
optimize(C, obj, []);
value(lambda)

% lstar = value(lambda);
% mu = dual(C(2));
% ustar = mu(3:4)/mu(1);
% value(ustar)
```

## GPU distance computation

### AABB包围盒的距离上下界

### BVTT的扩展
每个BVTT存储来自两颗BVH树中的两个节点，并在自适应深度下，生成自己的四个bvtt子节点，计算其bounding box的距离上下界，然后根据这个距离上下界，决定是否需要进一步的深度计算。

选择使用SoA的方式存储BVTT，这样可以更好的利用内存的连续性，提高访存效率
```cpp
__host__ __device__ struct g_bvtt{
	int* id1;
	int* id2;
	float* min;
};
```

### 自适应深度计算
对于一个有n个BVTT的buffer，需要计算其最大可扩展深度k，使其满足$2^{2k}n=C$，当k小于1时，取k=1，即采取传统策略
根据节点数对扩展的层数进行动态的调整，当缓冲区包含了n个BVTT节点时，自适应展开算法的目标通常是一个常数c，我们需要找到一个合适的展开层数k，在实际应用中，取常数c为$1024\times256$

```cpp
__host__ __device__ int calProDeep(int maxDeepA, int maxDeepB, int bvttLength, int deepNow)
```

### Parallel Culling
在每次迭代中，我们计算得到的每个BVTT节点中两个BVH节点的边界框之间的最小最大距离。随后，我们执行一个缩减操作来确定这些最大值中的最小值。然后将这个最小值作为最小距离的估计值。在后续的迭代中，所有边界框距离超过这个估计距离的BVTT节点都可以被有效地消除。这种筛选机制确保了在整个算法中对BVTT节点的控制和优化进程。

### 吞吐量
当buffer中的BVTT节点过少时，如何充分利用硬件的计算资源
传统的遍历方法通常为每个现有的BVTT节点分配一个线程，每个线程负责单个BVTT节点的扩展。虽然这种方法是标准的，特别是在前面描述的消除算法的距离计算中，但它往往没有充分利用GPU的资源，因为在每个内核调用中只处理有限数量的线程。

分配一个bvtt节点到多个对应线程中
考虑到一个bvtt节点最多有4个对应的子节点，
将一个BVTT节点分配到多个线程中，

```cpp
const int tid = blockDim.x * blockIdx.x + threadIdx.x;
int id1, id2;
id1 = bvttNode.id1[tid >> 2];
id2 = bvttNode.id2[tid >> 2];
```

### 最小距离的维护
然后使用cub库的BlockReduce原子操作获得一个block内的最小值，然后使用atomicMin原子操作获得全局的最小值
```cpp
// 基于浮点数的atomicMin操作
inline __device__ float atomicMin(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}
```

### BVH的重新构建
Why？ 对BVH树，需要进行调整，使其可以快速搜索到自己的所有第k层的后代节点

1.基于Mortan Code构建一颗完整二叉的BVH，其叶子节点包含一个或者两个primitive

2.而且相邻线程都是访问相同的或者相邻的BVTT，从而使得gpu上的内存访问基本是合并内存访问