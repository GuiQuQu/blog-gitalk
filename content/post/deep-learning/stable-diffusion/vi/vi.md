---
title: "变分推断"
description: 
date: 2023-04-20T19:59:05+08:00
image: 
math: true
comments: false
draft: true
url: /deep-learning/vi
categories:
 - Stable diffusion
---


# 变分推断

**背景**

求解后验$P(\theta|x)$成为推断，然后我们可以使用这个后验来进行决策

以贝叶斯推断为例
$$
\begin{align}
  P(\theta |X) & = \frac{P(X|\theta)P(\theta)}{P(X)} \\\
 & = \frac{ \text{似然} \cdot \text{先验} }{\int_\theta P(X| \theta)*P(\theta)d\theta} \\\
\end{align} \tag 8
$$
我们求解得到这个后验以后，可以用以下的方法来决策

给定一个新的样本$\bar{x}$,我们已知训练样本$X$,那么我们可以通过求解$P(\bar{x}|X)$来决策。其中，$\bar{x}$和$X$是通过$\theta$联系起来的。当给定$\theta$之后，$\bar{x}$和$X$之间相互独立。
$$
\begin{align}
p(\bar{x}|X) & = \int_\theta P(\bar{x},\theta|X)d\theta \\\
& = \int_{\theta} P(\bar{x}|\theta,X)P(\theta|X) d\theta \\\
& = \mathbb{E}_{\theta|x} [p(\bar{x}|\theta)] \\\
\end{align} \tag 9
$$


inference分类两类,一类是精确推断，就是我们可以求出后验或者是期望的值，第二类是近似推断，我们寻找近似值，近似推断又分为两类，确定性近似和随机近似。

变分推断(Variational Inference)是一种确定性近似

推导

X：observed data

Z：latent variable + parameter

（X，Z）：complete data

这里仍然要用到将似然写成ELBO + KL 散度的方式

$$
\begin{align}
& \log P(X) =  \log P(X,Z) - \log P(Z|X) \\\
& = \log \frac{P(X,Z)}{q(Z)} - \log \frac{P(Z|X)}{q(Z)}\\\
& 对于左右两侧同时求期望\\\
& \int_Z \log P(X)q(Z)dz = \log P(X) \\\
& = \int_Z (\log \frac{P(X,Z)}{q(Z)} - \log \frac{P(Z|X)}{q(Z)} )q(Z)dz \\\
& = \int_z q(Z) \log \frac{P(X,Z)}{q(Z)}dz - \int_Z q(Z)\log  \frac{P(Z|X)}{q(Z)}dz \\\
& = ELBO + KL(q||p) \\\
& = L(q) + KL(q || p) \\\
\end{align} \tag {10}
$$
$L(q)$称为变分，我们的目的是找到一个$q(Z)$,使得$q(Z)$和$P(Z|X)$近似，当两者越接近的时候，$KL(q \parallel p)$越接近于0，然后这个问题就变为了，我们要找一个$q(Z)$,使得$L(q)$达到最大值。
$$
\hat{q} = \arg \mathop{\max}\limits_{q(z)} L(q)
$$
求解

先做出假设，假设$q(Z)$可以划分成m个相互独立的变量组成(平均场理论)
$$
q(Z) \prod_{i=1}^m q_i(z_i)
$$
将上式带入，然后固定住只剩一个$q_j$，求解这个$q_j$，用这种方式将所有的$q_j$求出来

结论是下面这个

$L(q) = \sum_{i=1}^m KL(q_i(z_i)||\hat{P}(X,z_i))$
