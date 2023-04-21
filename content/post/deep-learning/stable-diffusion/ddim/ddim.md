---
title: "DDIM"
date: 2023-04-20T17:24:55+08:00
draft: false
url: /stable-diffusion/ddim
categories:
  - Stable Diffusion
---

DDIM相比于DDPM，不在假设扩散过程是一个马尔可夫链。

# DDPM回顾

**列一些比较重要的公式**

1. 扩散过程从$x_{t-1}$到$x_t$
   $$
   q(x_t | x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t\bold{I}) \tag 1
   $$
   使用重参数化技巧
   $$
   x_t = \sqrt{\beta_t} \epsilon + \sqrt{1- \beta_t} x_{t-1} \tag 2
   $$
   扩散过程
   $$
   q(x_{1:T}|x_0) =\prod_{t=1}^T q(x_t|x_{t-1}) \tag 3
   $$
   从$x_0$直接到$x_t$
   $$
   q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, 1-\bar \alpha_t \bold{I}) \tag 4
   $$
   使用重参数化技巧
   $$
   x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar \alpha_t} \epsilon \tag 5
   $$

2. 逆扩散过程

   从$x_{t}$到$x_{t-1}$假设为高斯分布
   
   $$
   p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t),\Sigma_\theta(x_t,t)) \tag 6
   $$
   
   逆扩散过程
   
   $$
   p_\theta(x_{0:T}) = p(x_T) \prod_{t=T}^1 p_\theta(x_{t-1}|x_t) \tag 7
   $$
   
   $q(x_{t-1}|x_t,x_0)$是一个高斯分布
   
   方差为
   
   $$
   \sigma^2_t = \frac{1- \bar \alpha_{t-1}}{1- \bar \alpha_t} \beta_t \tag 8
   $$
   

   均值为
   
   $$
   \widetilde \mu(x_t,x_0) = \frac{\sqrt{\alpha_t}(1-\bar \alpha_{t-1})}{1-\bar \alpha_{t}}x_t + \frac{\sqrt{\bar \alpha_{t-1}} \beta_t}{1-\bar \alpha_{t}}x_0 \tag 9
   $$
   
   $x_0$和$x_t$之间满足式(5),将$x_0$换为$x_t$
   
   $$
   \widetilde \mu(x_t) = \frac{1}{\bar \alpha_t}(x_t - \frac{\beta_t}{1-\bar \alpha_t}\epsilon) \tag{10}
   $$
   
   损失与$KL(q(x_{t-1}|x_t,x_0) \parallel p_\theta(x_{t-1}|x_t))$有关，两者均是高斯分布，q的方差固定，DDPM将p的方差也固定，因此参数只在均值中，高斯分布之间的KL散度有固定的公式，代换结束之后
   
   $$
      L_{t-1} = \mathbb{E}_q [\cfrac{1}{2 \sigma^2_t} \parallel \widetilde{\mu}_t(x_t,x_0) - \mu _ \theta (x_t,x_t) \parallel ^2] + C \tag{11}
   $$


   $$
   \begin{align}
   \mu_\theta(x_t,t)
   & = \frac{\sqrt{\alpha_t}(1-\overline \alpha_{t-1})}{1-\overline \alpha_{t}}x_t + \frac{\sqrt{\overline \alpha_{t-1}} \beta_t}{1-\overline \alpha_{t}}x_0^{\theta}\\\
   & = \frac{1}{\sqrt{\alpha_t}} (x_t - \frac{\beta_t}{\sqrt{1-\overline \alpha_t}} \epsilon_\theta (x_t,t)) \\\
   & = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\overline \alpha_t}}\epsilon_\theta  (\sqrt{\overline \alpha_t} x_0 + \sqrt{1-\overline \alpha_t} \epsilon,t)) \\\
   \end{align} \tag{12}
   $$

   $L_{t-1}$的新形式

   $$
   L_{t-1} = \mathbb{E}_{x_0,\epsilon} \Bigg[\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar \alpha_t)} \parallel \epsilon - \epsilon _\theta  (\sqrt{\bar \alpha_t} x_0 + \sqrt{1-\bar \alpha_t} \epsilon,t) ||^2 \Bigg] \tag {13}
   $$

去掉系数，得到$L_{simple}$

$$
L_{simple} = \mathbb{E}_{x_0,\epsilon} \parallel \epsilon - \epsilon _\theta  (\sqrt{\overline \alpha_t} x_0 + \sqrt{1-\overline \alpha_t} \epsilon,t) \parallel ^2 \tag{14}
$$

# DDIM

在DDIM中，符号代表的含义会改变，$\bar \alpha$ 使用$\alpha$表示

我们可以发现，在计算$L_{t-1}$的时候，我们只使用了$q(x_{t-1}|x_t,x_0)$和$p_\theta(x_{t-1}|x_t)$