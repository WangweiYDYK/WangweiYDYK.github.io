+++
title = 'Cuda Optimization'
date = 2022-12-03T18:26:30+08:00
tags = ['GPU']
categories = ["GPU"]
draft = true
+++

# Cuda代码优化

## 优化方向

| 优化方向             |
| -------------------- |
| Divergent branching  |
| bank conflicts       |
| memory coalescing    |
| Latency hiding       |
| Instruction overhead |

## 1.循环展开
目的：
    减少指令消耗
    增加更多的独立调度指令
    
```cpp

for (int i=0;i<100;i++)
{
    a[i]=b[i]+c[i];
}

// 循环展开减少遍历
for (int i=0;i<100;i+=4)
{
    a[i+0]=b[i+0]+c[i+0];
    a[i+1]=b[i+1]+c[i+1];
    a[i+2]=b[i+2]+c[i+2];
    a[i+3]=b[i+3]+c[i+3];
}
```

对于小循环，可以使用`#pragma unroll`来展开循环,因为使用编译器在编译的过程中会将确定的量优先存储在寄存器中，而SM中寄存器大小有限，所以适用于小循环

```cpp
__global__ void kernel(float * buf)
{
    float a[5];
    ...
    float sum = 0.0f;
    #pragma unroll
    for(int i = 0; i < 5; ++i)
        sum += a[i];
    ...
}
```

## 2.避免shared memory bank conflict

## 3.指令优化

整数除法和模运算的成本很高，因为它们最多可编译为20条指令。 在某些情况下，可以用位运算代替除法和取模运算：如果n是2的幂，则 `(i/n)` 等价于 `(i>>log2(n))` 并且 `(i%n)` 等价于 `(i&(n- 1))`; 如果 n 是字母，则编译器会执行这些转换。

倒数平方根：单精度的倒数平方根应该显式的调用`rsqrtf()`，双精度的应该调用`rsqrt()`

避免自动的双精度到单精度的转换。在以下两种情况下，编译器将时不时的插入转换指令，增加额外的执行周期：
a 函数操作char或者short时，其值通常要转换到int
b 双精度浮点常量(没有后缀的浮点数如：1.0是双精度，1.0f是单精度)，作为输入进行单精度浮点计算

4.使用`__restrict__`关键字来告诉编译器，指针指向的内存区域不会重叠。这样编译器就可以进行一些优化，比如不用考虑指针的别名问题。

## 线程束洗牌指令

线程束内交换变量\
`int __shfl(int var,int srcLane,int width=warpSize);`
`int __shfl_up(int var,unsigned int delta,int with=warpSize);`
`int __shfl_down(int var,unsigned int delta,int with=warpSize);`
`int __shfl_xor(int var,int laneMask,int with=warpSize);`

## 6.访存优化

合并访问：
    当一个线程束内的线程访问的内存都在一个内存块里的时候，此时为合并访问，而非合并访问会造成带宽浪费。
对齐访问：
    当一个内存事务的首个访问地址是缓存粒度的偶数倍的时候：此时为对齐内存访问，而非对齐内存访问会造成带宽浪费。

```cpp
int tx = threadIdx.x;
int ty = threadIdx.y;
__shared__ float buf[32][32];
float sum = 0.0f;

for (int j = 0 ; j < 32 ; j++) {
    sum += buf[j][tx];
}

for (int j = 0 ; j < 32 ; j++) {
     sum += buf[j][ty];
}

for (int j = 0 ; j < 32 ; j++) {
    sum += buf[ty][j];
}

for (int j = 0 ; j < 32 ; j++) {
     sum += buf[tx][j];
}
```

## 7.避免寄存器溢出

```cpp
__global__ void
__launch_bounds__(maxThreadaPerBlock,minBlocksPerMultiprocessor)
kernel()
{

}
```

## 8.常量内存的读取

## 9.避免统一线程束内的线程分化

<span style="color:red;">
    当一个线程束中所有的线程都执行if或者，都执行else时，不存在性能下降；只有当线程束内有分歧产生分支的时候，性能才会急剧下降。</span>

## 内存的传输

对于多个小规模的数据传输，最好将其合并为单独的一次数据传输。

对于二位数组的传输，可以使用cudaMemcpy2D()函数，这样可以减少内存的传输次数。

## Nsight的使用

### atomic operation

```cpp
	__device__ int atomicAggInc(int *ctr) {
		int mask = __ballot_sync(__activemask(), 1), leader, res;
		// select the leader
		leader = __ffs(mask) - 1;
		// leader does the update
		if (lane_id() == leader)
			res = atomicAdd(ctr, __popc(mask));
		// broadcast result
		res = warp_bcast(res, leader);
		// each thread computes its own value
		return res + __popc(mask & ((1 << lane_id()) - 1));
	}
```