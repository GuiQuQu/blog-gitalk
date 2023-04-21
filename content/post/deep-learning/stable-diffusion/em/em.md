---
title: "EM-Algorithm"
description: 
date: 2023-04-20T19:59:21+08:00
image:
url: /deep-learning/em
math: true
comments: false
draft: false
categories:
 - Deep learning
---

# 贝叶斯公式

$$
\begin{aligned}
& P(x|y) = \frac{P(x,y)}{P(y)} = \frac{P(y|x) p(x)}{P(y)} \\\
& P(x,y) = P(x|y)P(y) \\\
& P(y) = \frac{P(x,y)}{P(x | y)} \\\
\end{aligned}
$$

$$
\begin{aligned}
& P(A,B,C) = P(C|B,A)P(B,A) = P(C|B,A)P(B|A)P(A) \\\
& P(B,C|A) = \frac{P(A,B,C)} {P(A)} = P(C|A,B) P(B|A) \\\
\end{aligned}
$$

# 隐变量模型-EM算法

我们只能观察到变量$X$，那么我们可以采用MLE来直接求解对数极大似然，公式如下，$\theta$是模型参数
$$
 \theta_{MLE} = \arg \mathop{\max}\limits_{\theta} \log P(X | \theta)
$$
如果我们确定一个隐变量模型，一般使用$z$表示隐变量，隐变量$z$服从一个分布，条件概率$P(X|给确定的z)$服从另一种分布。

但是对于隐变量模型来说，直接求解MLE根本无法计算，因此EM算法就是用来解决含有隐变量模型的极大似然估计问题

EM算法分为E-step和M-step

**EM公式**
$$
\begin{aligned}
& \theta^{(t+1)} = \arg \mathop{\max}\limits_{\theta} \int_{z} \log P(x,z | \theta) \cdot P(z|x,\theta^{(t)}) dz \\\ 
\end{aligned} \tag{1}
$$

$$
右边 = \arg \mathop{\max}\limits_{\theta} \int_{z} \log P(x,z | \theta) \cdot P(z|x,\theta^{(t)}) dz =\arg \mathop{\max}\limits_{\theta} E_{z|x,\theta^{(t)}} [ \log P(x,z|\theta) ] 
$$

算法收敛性证明（不完整，证明(2)式成立,保证我们迭代的结果一定会变好）

$$
log P(X | \theta^{(t)}) \le log P(X | \theta^{(t+1)}) \tag{2}
$$

证明如下

$$
\begin{aligned}
& \log P(x|\theta) = \log \frac{P(x,z |\theta)} {P(z|x,\theta)} = \log P(x,z |\theta) - \log P(z|x,\theta) \\\
& 公式两侧关于z|x,\theta^{(t)},求积分，得到期望 \\\
& 左边 = \int_z \log P(z|\theta) P(z|x,\theta^{(t)})dz \\\
&  = \log P(x|\theta) \int_z  P(z|x,\theta^{(t)})dz \\\
& = \log P(x|\theta) * 1 \\\
& 右边 = \log P(x,z |\theta) - \log P(z|x,\theta) \\\
& = \int_z \log P(x,z|\theta) P(z|x,\theta^{(t)})dz - \int_z \log P(z|x,\theta) P(z|x,\theta^{(t)})dz \\\
& = Q(\theta,\theta^{(t)}) - H (\theta,\theta^{(t)}) \quad \leftarrow 定义的符号\\\
\end{aligned}
$$

其中$Q(\theta,\theta^{(t)})$即为$(1)$中求解的期望，我们可以确定的得到$Q(\theta^{(t+1)},\theta^{(t)}) \ge Q(\theta,\theta^{(t)})$,也就有$Q(\theta^{(t+1)},\theta^{(t)}) \ge Q(\theta^{(t)},\theta^{(t)})$

当我们在证得$H(\theta^{(t)},\theta^{(t)}) \ge H(\theta^{(t+1)},\theta^{(t)})$,我们就可以得到(2)式成立
$$
\begin{align}
H(\theta^{(t)},\theta^{(t)}) - H(\theta^{(t+1)},\theta^{(t)}) & = \int_z \log P(z|x,\theta^{(t)}) P(z|x,\theta^{(t)})dz - \int_z \log P(z|x,\theta^{(t+1)}) P(z|x,\theta^{(t)})dz   \\\
& = \int_z  P(z|x,\theta^{(t)}) \log \frac{P(z|x,\theta^{(t)})}{P(z|x,\theta^{(t+1)})} dz \\\
& = - KL(P(z|x,\theta^{(t)}) \parallel P(z|x,\theta^{(t+1)})) \\\
\end{align} \tag{3}
$$

因为KL散度的公式的公式是交叉熵减熵，然后熵自带负号。因为KL散度一定是 >= 0的,因此$(3) \le 0$ 成立,从而$(2)$式得证。这里我们也可以直接使用[Jensen不等式](https://zhuanlan.zhihu.com/p/39315786)求解(log函数是凸函数)



**EM公式**

E-Step,求解期望
$$
P(z|x,\theta^{(t)}) \rightarrow E_{z|x,\theta^{(t)}} [\log P(x,z|\theta)] \tag 4
$$


M-Step，最大化期望
$$
\theta^{(t+1)} = \arg \mathop{\max}\limits_{\theta}  E_{z|x,\theta^{(t)}} [\log P(x,z|\theta)] \tag{5}
$$
公式推导,和上面的证明是类似的
$$
\begin{aligned}
& \log P(x|\theta) = \log \frac{P(x,z |\theta)} {P(z|x,\theta)} = \log P(x,z |\theta) - \log P(z|x,\theta) \\\
& \\\
& 引入 关于z的分布q(z) \\\
& \log P(x|\theta) = \log \frac{P(x,z |\theta)}{q(z)} - \log \frac{P(z|x,\theta)}{q(z)} \\\
& 等式两边同时关于q(z)求期望 \\\
& 左边 = \int_z \log P(z|\theta) * q(z) dz \\\
&  = \log P(x|\theta) \int_z  q(z) dz \\\
& = \log P(x|\theta) * 1 \\\
& 右边 = \int_z q(z) \log \frac{P(x,z |\theta)}{q(z)} - \int_z q(z)  \log \frac{P(z|x,\theta)}{q(z)} \\\
& =ELBO + KL(q(z)||P(z|x.\theta)) \\\
& = Q(\theta,\theta^{(t)}) - H (\theta,\theta^{(t)}) \quad 定义的符号\\\
\end{aligned} \tag{6}
$$

其中$ELBO$(evidence lower bound)

$$
\begin{aligned}
& \log P(x|\theta) = ELBO + KL(q||p) \\\
& \Rightarrow \log P(x | \theta) >= ELBO
\end{aligned}
$$
所以ELBO是一个极大似然值的下界，当前q和p的分布相同时，此时$\log P(x|\theta) = ELBO$

EM算法是在逐渐使得ELBO变大，使得对数似然值变大
$$
\begin{align}
\theta ^{(t+1)} & = \arg \mathop{\max}\limits_{\theta} ELBO \\\
& =\arg \mathop{\max}\limits_{\theta} \int_z q(z) \log \frac{P(x,z|\theta)}  {q(z)} \\\
& \text{当} q(z) =P(z|x,\theta^{(t)}) \text{的时候,ELBO和似然值相等(使得期望最大)} \\\
& = \arg \mathop{\max}\limits_{\theta} \int_z P(z|x,\theta^{(t)}) \log \frac{P(x,z|\theta)}{P(z|x,\theta^{(t)})} \\\
& =\arg \mathop{\max}\limits_{\theta} \int_z P(z|x,\theta^{(t)}) [\log P(x,z|\theta) - \log  P(z|x,\theta^{(t)})] \\\ 
& \text {后一项中没有一个数值和}\theta \text{有关，因此是一项常数,可以直接去掉,得到} \\\
& =\arg \mathop{\max}\limits_{\theta} \int_z P(z|x,\theta^{(t)}) \log P(x,z|\theta)
\end{align} \tag{7}
$$
从而得到EM算法的核心迭代公式