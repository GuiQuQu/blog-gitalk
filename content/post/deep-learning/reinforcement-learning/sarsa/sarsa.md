---
title: "Sarsa算法"
description: 
date: 2023-06-10T00:37:00+08:00
image:
url: /rl/sarsa/
math: true
comments: false
draft: false
categories:
    - rl
---

目前强化学习相关的内容全部来自于[DRL](https://github.com/wangshusen/DRL)
我也只是把一些基本的概念写下来，防止自己忘记了

# sarsa算法

## TD target推导

我们已知折扣回报的定义
$$
U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots + \gamma^{n-t} R_n
$$
进行简单的变化,可以得到
$$
\begin{aligned}
U_t  & = R_t + \gamma (R_{t+1} + \gamma^1 R_{t+2} + \cdots + \gamma^{n-t-1} R_n)  \\
 & = R_t + \gamma U_{t+1}
\end{aligned}
$$

即有
$$
\ U_t = R_t + \gamma U_{t+1}
$$

假定$R_t$依赖于$S_t,A_t,S_{t+1}$
我们可以得到

$$
\begin{aligned}
Q_\pi(s_t,a_t) & = \mathbb{E}[U_t | s_t,a_t] \\
& = \mathbb{E}[R_t + \gamma U_{t+1} | s_t,a_t] \\
& = \mathbb{E}[R_t|s_t,a_t] + \gamma \mathbb{E}[U_{t+1} | s_t,a_t] \\
& = \mathbb{E}[R_t|s_t,a_t] + \gamma \mathbb{E}[Q_\pi(S_{t+1},A_{t+1})|s_t,a_t] \\
\end{aligned}
$$

从而我们可以得到

$$
Q_\pi(s_t,a_t) = \mathbb{E}_{S_{t+1},A_{t+1}}[R_t + \gamma Q_\pi(S_{t+1},A_{t+1})] \ \text{for all} \ \pi
$$

实际上期望很难求,因此我们采用Monte Carlo方法来估计期望,即使用采样值来代替期望值。
因此我们可以得到

$$
Q_\pi(s_t,a_t) \approx r_t + \gamma Q_\pi(s_{t+1},a_{t+1}) \ \text{for all} \ \pi
$$

等式右侧就是 **TD target** $y_t = r_t + \gamma Q_\pi(s_{t+1},a_{t+1})$。

TD learning 鼓励$Q_\pi(s_t,a_t)$尽可能地接近TD target $y_t$。因为$y_t$有一部分是真实值,比前一步的随机乱猜结果要好，这种方式是自举的，因为可以使用自己更新自己。

## TD learning for Value Network

- 将$Q_\pi(s,a)$使用价值网络近似为$q(s,a;\bold{w})$，其中$\bold{w}$是神经网络的参数。
- $q$ 也在actor-critic中使用，职责是crr'tic，即评价actor的表现。
  

**TD error & Gradient**

- TD target: $y_t = r_t + \gamma q(s_{t+1},a_{t+1};\bold{w})$
- TD error: $\delta_t = q(s_t,a_t;\bold{w}) - y_t$
- Loss: $L(\bold{w}) = \frac{1}{2} \delta_t^2$
- Gradient: $\nabla_{\bold{w}} L(\bold{w}) = \delta_t \nabla_{\bold{w}} q(s_t,a_t;\bold{w})$
- Update: $\bold{w} \leftarrow \bold{w} - \alpha \delta_t \nabla_{\bold{w}} q(s_t,a_t;\bold{w})$
  
我们采用梯度下降更新参数。其中$\alpha$是学习率。

# Reference

1. [DRL](https://github.com/wangshusen/DRL)