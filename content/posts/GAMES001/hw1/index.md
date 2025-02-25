+++
title = 'Abstract Algebra'
date = 2024-02-26T19:08:59+08:00
draft = false
tags = ['math']
categories = ["math"]
+++
# hw1

## Q2

### a
$a\circ b\in G$。
$$\mathbf{A}x = \lambda x$$
$$\mathbf{A}^n x = \lambda ^ n x$$
$$ k\mathbf{A}^n x = k \lambda ^ n x$$
$$(c_k \\mathbf{A}^k + c_{k-1} \\mathbf{A}^{k-1} + \cdots + c_1 \mathbf{A} + c_0 \mathbf{I})x = (c_k \lambda^k + c_{k-1} \lambda^{k-1} + \cdots + c_1 \lambda + c_0 )x$$
即有 f(λ) 为 f(A) 的特征值。
### c

$$e^{\mathbf{A}}= \sum_{k=0}^{\infty} \frac{1}{k!} \mathbf{P}^{-1}\Lambda^{k} \mathbf{P}= \mathbf{P}^{-1}(\sum_{k=0}^{\infty} \frac{1}{k!}\Lambda^{k}) \mathbf{P}$$

$$e^{\mathbf{A}^T}= \sum_{k=0}^{\infty} \frac{1}{k!} \mathbf{P}^{T}\Lambda^{k} \mathbf{P}^{-1^T}= \mathbf{P}^{T}(\sum_{k=0}^{\infty} \frac{1}{k!}\Lambda^{k}) \mathbf{P}^{-1^T}$$

由于 $\Lambda$ 为对角矩阵，所以 $\Lambda^k$ 也是对角矩阵，所以 $\sum_{k=0}^{\infty} \frac{1}{k!}\Lambda^{k}$ 也是对角矩阵。c问易证

### d
$$e^{\mathbf{A}}= \sum_{k=0}^{\infty} \frac{1}{k!} \mathbf{P}^{-1}\Lambda^{k} \mathbf{P}= \mathbf{P}^{-1}(\sum_{k=0}^{\infty} \frac{1}{k!}\Lambda^{k}) \mathbf{P}$$

$$det(e^{\mathbf{A}}) = det(\sum_{k=0}^{\infty} \frac{1}{k!} \Lambda^{k}) = e^{\lambda_1}e^{\lambda_2}...e^{\lambda_n-1}e^{\lambda_n} = e^{Tr{\mathbf{B}}}$$

### e
$$\mathbf{P} = ba^{T}ab^{T}+ab^{T}ba^{T}-ab^{T}ab^{T}-ba^{T}ba^{T}$$

### f