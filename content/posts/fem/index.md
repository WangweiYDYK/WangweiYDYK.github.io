+++
title = 'FEM'
date = 2023-07-13T19:08:59+08:00
draft = false
+++

# fem概览

## 流程

1.获得微分方程  
2.定义边界或者约束条件  
3.将微分形式的控制方程转换为其等效积分  
4.对计算单元刚度矩阵  
5.单元刚度矩阵的组装  
6.矩阵求解  

## 等效变换
控制方程（微分形式）-> 等效积分 -> 等效积分的弱形式

变分形式推导


## 加权余量

给定微分方程
$$A(u) = 0$$
$$B(u) = 0$$

能量泛函推导：
$$A(u) = L(u) + f$$
$$\int_{\Omega}(L(u) + f)\delta u \  d\Omega = 0$$

$$ \begin{equation*}
\begin{split}
    \int_{\Omega}L(u)\delta u \  d\Omega
    & = \int_{\Omega}\frac{1}{2}L(u)\delta u \  d\Omega + \int_{\Omega}\frac{1}{2}L(u)\delta u \  d\Omega \\\
    & = \int_{\Omega}\frac{1}{2}L(u)\delta u \  d\Omega + \int_{\Omega}\frac{1}{2}L(\delta u)u \  d\Omega \\\
    & = \delta\int_{\Omega}\frac{1}{2}L(u)u \  d\Omega
\end{split}
\end{equation*} $$
其中A和B都是算子符号,$A(u)$为控制方程，$B(u)$为边界条件  
其等效积分形式为
$$\int wA(u)d\Omega + \int wB(u)d\tau = 0$$
当通过形函数进行近似时
$$\tilde{u} = \sum N_iu_i$$
$$A(\tilde{u}) = R$$

>Example:


### 伽辽金法

用单元的形函数来代表等效积分中的权函数$w$使得残差尽可能小
$$\int N A(\tilde{u})d\Omega + \int NB(\tilde{u})d\tau = 0$$

### 其他方法

子域法，配点法，最小二乘法，力矩法

## 边界条件
对一个2m阶的微分方程，0到m-1阶为强制边界条件，m到2m-1为自然边界条件  
强制边界条件（本质边界条件）：强加给控制方程必须满足的  
自然边界条件：泛函一阶变分为零，在边界上必须满足的条件（一般在积分表达式中可以自动得到满足）  
混合边界条件

### Dirchlet边界条件

#### 常微分条件下

在区间$\left [ a,b \right ]$,满足$y(a) = \alpha, y(b) = \beta$,其中$\alpha,\beta$为常数

#### 偏微分条件下
$y(x) = f(x), \forall x \in \partial \Omega$,其中 $f$ 是在边界 $\partial \Omega$ 中定义的已知函数

Example：

    机械：梁的一端保持在空间中的固定位置  
    热力学中：表面保持在固定温度
    流体力学：粘性流体的固液边界处，流体相对于边界具有零速度

### Neumman边界条件

代求变量边界外法线的方向导数被指定

#### 常微分条件下

在区间$\left [ a,b \right ]$,满足$y'(a) = \alpha, y'(b) = \beta$,其中$\alpha,\beta$为常数

#### 偏微分条件下

Example：

    热力学：热传导方程中边界绝热，内部热量无法通过边界传导到外部

### Robin边界条件

## 施加约束

### 罚函数

P（比例）控制器  
PD（比例微分）控制器  
PID（比例积分微分）控制器  

### 拉格朗日乘子法

## 函数内积
$\int_{a}^{b}f(x)g(x)dx$记作$\left \langle f,g \right \rangle$称为函数内积

若$\left \langle f,g \right \rangle$在$\left [ a,b \right ]$上等于0，说明$\left \langle f,g \right \rangle$在$\left [ a,b \right ]$上正交


## 单元刚度矩阵的推导

### 偏微分法

### 变分法

## 静态分析中平衡方程求解

### 直接求解

### 迭代求解
    Gauss-Seidel方法

### 非线性方程组求解
    Newton-Raphson方法
    BFGS法
    载荷-位移-约束方法

## 动态分析中平衡方程求解

### 直接积分
    中心差法
    Houbolt法
    Newmark法
    Bathe法


### 模态叠加

## 与FVM，FDM的差别

对于FVM其权函数$w$为1

## 线性动态有限元

由于引入了时间坐标，因此问题变为二维$(x, t)$问题，采用部分离散的方法，即只将空间域进行离散

## 弹性体模拟

