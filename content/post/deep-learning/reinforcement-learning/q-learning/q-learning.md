---
title: "Q-Learning"
description: 
date: 2023-06-10T00:37:21+08:00
image:
url: /rl/q-learning/
math: true
comments: false
draft: false
categories:
    - rl
---

# Q-learning

## Sarsa vs. Q-Learning

**Sarsa:**

- Sarsa 用来训练动作价值函数$Q_\pi(s,a)$
- TD target: $y_t = r_t + \gamma Q_\pi(s_{t+1},a_{t+1})$
- Sarsa 用来更新价值网络(critic)

**Q-learning:**

- Q-learning 用来训练最优动作价值函数$Q^*(s,a)$
- TD target $y_t = r_t + \gamma \max_{a_{t+1}} Q^*(s_{t+1},a_{t+1})$
- 使用Q-learning更新DQN

## Q-learning TD target

我们已经在Sarsa算法中看到下式成立
$$
Q_\pi(s_t,a_t) = \mathbb{E}_{S_{t+1},A_{t+1}}[R_t + \gamma Q_\pi(S_{t+1},A_{t+1})] \ \text{for all} \ \pi
$$
因此对于最优策略$\pi^*$,上式也同样成立，即有

$$
Q^*(s_t,a_t) = \mathbb{E}_{S_{t+1},A_{t+1}}[R_t + \gamma Q^*(S_{t+1},A_{t+1})] 
$$

动作$A_{t+1}$通过下式计算
$$
A_{t+1} = \arg \max_{a} Q^*(S_{t+1},a)
$$
因此，我们可以得到
$$
Q^*(S_{t+1},A_{t+1}) = \max_{a} Q^*(S_{t+1},a)
$$
带入最优策略的Bellman方程中，我们可以得到
$$
Q^*(s_t,a_t) = \mathbb{E}_{S_{t+1}}[R_t + \gamma \max_{a} Q^*(S_{t+1},a)]
$$

然后采用Monte Carlo方法来估计期望，即使用采样值来代替期望值。因此我们可以得到
$$
Q^*(s_t,a_t) \approx r_t + \gamma \max_{a} Q^*(S_{t+1},a)
$$
等式右侧就是 **TD target** $y_t = r_t + \gamma \max_{a} Q^*(S_{t+1},a)$。

## Q-learning训练DQN

同样采用TD算法

- 使用神经网络$Q(s,a;\bold{w})$来近似$Q^*(s,a)$
- DQN通过
$$
  a_t = \arg \max_{a} Q(s_t,a;\bold{w})
$$
来选择动作控制Agent
- 可学习的参数为$\bold{w}$

对于一个可以观察到转移$(s_t,a_t,r_t,s_{t+1})$
- TD target： $y_t = r_t + \gamma \max_{a} Q(s_{t+1},a;\bold{w})$
- TD error: $\delta_t = Q(s_t,a_t;\bold{w}) - y_t$
- Loss: $L(\bold{w}) = \frac{1}{2} \delta_t^2$
- Update：$\bold{w} \leftarrow \bold{w} - \alpha \delta_t \cdot \frac{\partial Q(s_t,a_t;\bold{w})}{\partial \bold{w}}$

# Reference

1. [DRL](https://github.com/wangshusen/DRL)