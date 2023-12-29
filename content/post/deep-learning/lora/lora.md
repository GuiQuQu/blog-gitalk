---
title: "Lora 模型高效参数微调"
description: 
date: 2023-12-11T21:07:11+08:00
image:
url:
math: true
comments: true
draft: false
categories:
---

# 安装与使用

```shell
pip install peft
```

**使用最新特征**

```shell
pip install git+https://github.com/huggingface/peft.git
```

# Lora原理介绍

1. 在神经网络中会有很多矩阵相乘,一个非常典型的例子就是`nn.Linear`,内部的矩阵即为`nn.Linear`的参数(不考虑bias)

Lora的假设:在模型参数更新的过程,模型的参数矩阵是低"内在秩"的(low intrinscic rank)

因此Lora的想法使用一个低秩分解矩阵约束这个参数矩阵的更新

$$
 W_0 + \Delta W = W_0 + BA
$$

其中$W_0 \in \mathbb{R}^{d \times k}$是原始的参数矩阵,$B \in \mathbb{R}^{d \times r}$,$A \in \mathbb{R}^{r \times k}$是低秩的可分解矩阵。而且$r \ll  \min(d,k)$

$W_0 + \Delta W$按照全参数微调的理解,$\Delta W$就是模型参数的变换量,然后按照LoRA的假设,这个更新量是`low intrinscic rank`的,因此用低秩分解来表示它,从而达到降低参数量的目的。

在训练过程中,$W_0$冻结,只更新$B$和$A$。

# SVD分解

## 特征值与特征向量

首先回顾下特征值和特征向量的定义如下：
$$
    Ax = \lambda x
$$

其中$A$是一个$n \times n$矩阵， $x$是一个$n$维向量，则$\lambda$是矩阵 的一个特征值，而$x$是矩阵$A$的特征值$\lambda$所对应的特征向量。

## 矩阵的SVD分解定义

任何一个矩阵$A \in R^{m \times n}$ 都可以分解成$A = U \Sigma V^T$,这里$U \in R^{m \times m}$,$V \in R^{n \times n}$是正交矩阵,$\Sigma \in R^{m \times n}$是对角矩阵,除了主对角线上的元素以外全为0。

补充 
1. 正交矩阵的定义: $A A^T = E$,则称n阶方阵A为正交矩阵