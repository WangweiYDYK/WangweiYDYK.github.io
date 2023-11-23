---
layout: post
title: Sun-of-Square CCD
tags: simulation
math: true
date: 2023-11-23 15:32 +0800
---

# Sum-of-Squares Collision Detection

## Putinars Positivstellensatz Theorem
对于一个具有代数紧致性的域$\mathbb{D}=\left\{\mathbf{u} \in \mathbb{R}^k: \forall g \in \mathcal{G}, h \in \mathcal{H}, g(\mathbf{u}) \geq 0, h(\mathbf{u})=0\right\}$，任何在$\mathbb{D}$上严格正定的多项式函数$f(\mathbf{u})$都是$\mathcal{Q}\left(\mathcal{G},\mathcal{H} \right)_d$的一个元素，其中$Q(\mathcal{G}, \mathcal{H})_d=\left\{s_0+\sum_{g \in \mathcal{G}} s_g g+\sum_{h \in \mathcal{H}} p_h h: \begin{array}{l}s_0 \in \Sigma, s_g \in \Sigma_d \\ p_h \in \mathbb{R}[\mathbf{u}]_d\end{array}\right\}$

## Sum-of-Squares (SOS) Programming
对于问题
$$\begin{aligned} & f^*=\min _{\mathbf{u} \in \mathbb{D}} f(\mathbf{u}) \\ & \mathbf{u}^*=\underset{\mathbf{u} \in \mathbb{D}}{\arg \min } f(\mathbf{u}) \\ & \mathbb{D}=\left\{\mathbf{u} \in \mathbb{R}^k: g_i(\mathbf{u}) \geq 0, h_i(\mathbf{u})=0\right\}.\end{aligned}$$
$f^*=\gamma^*=\max \{\gamma: f(\mathbf{u})-\gamma$ is positive for $\mathbf{u} \in \mathbb{D}\}$.是一个NP-hard问题。由于$f(\mathbf{u}) - \gamma$为$Q(\mathcal{G}, \mathcal{H})_d$的元素。由Putinars Positivstellensatz Theorem可以将其relax为一个convex SDP即
$f_d^* = \max \{\gamma: f(\mathbf{u})-\gamma \in Q(\mathcal{G}, \mathcal{H})_d\}$
$f_d^* \le f^*$
$d \rightarrow \infty, f_d^* \rightarrow f^*$

### SOS FORM
$\lambda^*=\left\{\begin{array}{ll}\max _{\lambda \in \mathbb{R}} & \lambda \\ \text { s.t. } & f-\lambda-\sum_i h_i p_i-\sum_i g_i s_i \in \Sigma_d, \\ & s_i \in \Sigma_d \\ & p_i \in \mathbb{R}[\mathbf{u}]_d\end{array}\right\}$.

## Multiple Patches in One Optimization

若$\mathbb{D}_1=\left\{\mathbf{u}^1: g_i^1\left(\mathbf{u}^1\right) \geq 0\right\}$,$\mathbb{D}_2=\left\{\mathbf{u}^2: g_i^2\left(\mathbf{u}^2\right) \geq 0\right\}$,则其笛卡尔积(Cartesian product)为$\begin{aligned} \mathbb{D} & =\left\{\left(\mathbf{u}^1, \mathbf{u}^2\right): g_i^1\left(\mathbf{u}^1\right) \geq 0, g_i^2\left(\mathbf{u}^2\right) \geq 0, q_j\left(\mathbf{u}^1, \mathbf{u}^2\right) \geq 0\right\} \\ & \subseteq \mathbb{D}_1 \times \mathbb{D}_2,\end{aligned}$

## CCD
t时刻的quadratic and cubic Bezier triangle可以表示为
$x(u, v, t)=\sum_i^{n_B}\left(\mathbf{p}_i+\mathbf{v}_i t\right) \phi_i(u, v)$
![Alt text](Y1S$OY22%7B7Z4D@13V4B1BX6.png)

其CCD domain由$\mathbf{u} = $
碰撞约束$\mathcal{H}=\left\{x_1\left(u_1, v_1, 0\right)_{x y z}-x_2\left(u_2, v_2, 0\right)_{x y z}\right\}$

## 线性路径的CCD

## Curved Path CCD

### Dual Quaternion


## SOS Collision Detection Certificates

Intersecting Pair(IP)
Earliest Collision(EC)
Non-Collision(NC)