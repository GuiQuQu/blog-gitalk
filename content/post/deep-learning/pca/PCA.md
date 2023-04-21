---
title: "PCA"
description:
date: 2022-04-20T17:24:55+08:00
url: /deep-learning/pca
math: true
draft:  false
categories:
    - Deep Learning
---

# 问题描述

给定数据$X_{(N,d)} = ({x_1^T,x_2^T,x_3^T ... x_N^T})$

我们用n表示数据个数，d表示数据维度。并且每一条数据都是$(d \times 1)$的列向量

**总览**

为什么需要降维，因为维度灾难的问题，当数据维度越高时，分布越稀疏，因此我们需要更多的数据才能更好的表示数据实际分布，当数据过少时，我们使用的训练集就会出现较大的偏差，从而造成了过拟合现象。

PCA是一种线性降维方法，我们可以通过直接对数据矩阵$X$做奇异值分解或者是对数据方差矩阵做特征值分解来得到需要成的矩阵。

PCA的核心思想：将一组可能线性相关的向量变化到一组线性无关向量的表示。

有两种理解方式

- 最大投影方差
- 最小重构距离

# 铺垫

数据均值和方差

**均值**
$$
\bar{X}_{(d,1)} = \frac{1}{N} \sum_{i=1}^n x_i
$$
转换为采用矩阵表示的形式
$$
\bar{X} = \frac{1}{N} (x_1,x_2,...,x_n) (1,1,...,1)_{(N,1)}^T = \frac{1}{N} X^T 1_N
$$
记$(1,1,...,1)_{(N,1)}^T = 1_N$

求解均值的原因是我们需要将数据中心化，这样比较好算

方差/协方差
$$
S = \frac{1}{N}\sum_{i=1}^{N} (x_i - \bar{x})^2
$$
对于多维向量(分布来说)表示为协方差矩阵
$$
S = \frac{1}{N}\sum_{i=1}^{N} (x_i - \bar{x})(x_i - \bar{x})^T
$$
转换矩阵表达形式
$$
S = \frac{1}{N} X^THH^TX
$$
其中$H = E_{(N.N)} - \frac{1}{N} 1_N 1_N^T$，是中心矩阵，可用来将数据中心化$\bar{X} = H^TX$

展开的方式和均值类似，之后再写。

中心矩阵有以下性质

- $H^T = H$
- $H^2 = H$

所以说协方差可以简化成
$$
S = \frac{1}{N} X^THX
$$

# PCA-最大投影方差角度

我们拿出投影之后某一个基$\vec{u}$，对于原始的数据点来说，我们希望在投影到方向$\vec{u}$之后，各个数据点可以划分的最散，即投影之后方差最大。

首先我们先将数据进行变化，使得中心点为零点，这样计算方差容易计算 

对于一个点$\vec{x_i}$来说，中心化的结果是$\vec{x_i} - \bar{x}$

**计算投影**

对于任意一个$\vec{a}$，来说，因为$\vec{u}$是单位向量，因此投影到$\vec{u}$方向上的模长就是$\vec{a} \times \vec{u}$
$$
Prj_{\vec{u}} = (\vec{x_i} - \bar{x}) \times \vec{u} = (\vec{x_i} - \bar{x})^T u
$$


在算完投影之后，我们希望投影的方差最大，因此我们使用方差的公式
$$
\begin{aligned}
J & = \frac{1}{N} \sum_{i=1}^N ((x_i - \bar{x})^Tu)^2 \\
	& = \frac{1}{N} \sum_{i=1}^N  ((x_i - \bar{x})^Tu)^T (x_i - \bar{x})^Tu  \\
	& = \frac{1}{N} \sum_{i=1}^N u^T (x_i - \bar{x})(x_i - \bar{x})^Tu \\
	& = \frac{1}{N} u^T \sum_{i=1}^N (x_i - \bar{x})(x_i - \bar{x})^T u \\
	& = u^T S u \\
\end{aligned}
$$
从而我们可以得到完整的优化问题，带一个约束
$$
\begin{cases}
u = argmax \ u^TSu \\
 s.t. \ u^Tu = 1
 \end{cases}
$$
采用拉格朗日方法求解
$$
f(u,\lambda) = u^TSu + \lambda(u^Tu - 1)
$$
导数等于0得到
$$
Su = \lambda u
$$
从而我们发现了$u$实际上是$S$的一个特征向量，我们将该式带入$u^TSu = \lambda$，因此我们在取特征向量时应该按照特征值从大到小取。

# PCA-最小重构代价角度

 PCA可以看成对原始特征空间的重构，我们重新构造了$d$个线性无关的基$u_1,u_2,...,u_d$,使用新的坐标系表示

在经过重构之后，我们可以得到$x_i$的表示(先忽略中心化)
$$
x_i = \sum_{j=1}^d (x_i^Tu_j)u_j
$$
因为我们做的是降维，因此我们只选择p组基，如果全选就只是一种重构。

因此降维后
$$
\hat{x_i} = \sum_{j=1}^p (x_i^Tu_j)u_j
$$
我们希望最小化重构代价，那么考虑重构代价的表达式

对于一个点$x_i$，该点的重构代价为
$$
J = x_i - \hat{x_i}
$$
我们全部点重构代价的模之和最小
$$
\begin{aligned}
J & = \frac{1}{N} \sum_{i=1}^N \Vert x_i - \hat{x_i}  \Vert_2 \\
& = \frac{1}{N} \sum_{i=1}^N \sum_{j=p+1}^d \Vert (x_i^Tu_j)u_j \Vert_2 \\ 
& = \frac{1}{N} \sum_{i=1}^N \sum_{j=p+1}^d (x_i^Tu_j)^2
\end{aligned}
$$
在算上之前忽略的中心化，我们可以得到，按照和投影方差相同的变化范式
$$
\begin{aligned}
J & = \frac{1}{N} \sum_{i=1}^N \sum_{j=p+1}^d ((x_i-\bar{x})^Tu_j)^2 \\
& = \sum_{j=p+1}^d (\frac{1}{N} \sum_{i=1}^N u_j^T (x_i-\bar{x})(x_i-\bar{x})^Tu_j) \\
& = \sum_{j=p+1}^d ( u_j^T( \frac{1}{N} \sum_{i=1}^N (x_i-\bar{x})(x_i-\bar{x})^T)u_j\\
& = \sum_{j=p+1}^d u_j^T Su_j
\end{aligned}
$$
这样就得到最小重构代价，我们最小化重构代价
$$
\begin{cases}
u_j = argmin \sum_{j=p+1}^d \ u_j^TSu_j \\
 s.t. \ u_j^Tu_j = 1
 \end{cases}
$$
因为每一个$u_j$都是线性无关的，因此这个求和的最小化问题可以针对每一个$u_j$一个一个解，从而就得到了和最大投影方差相同的结论。

# PCA-SVD角度

我们直到了要算出投影结果就需要求解协方差矩阵的特征向量，那么我们按照特征值分解的方式

$S$是实对称阵，一定存在特征值分解，并且每个特征向量之间两两正交
$$
S = GKG^{-1} = GKG^T
$$
其中$G^TG =E$,$K$是特征值矩阵

我们先得到中心化之后的数据矩阵，为$HX$
$$
对数据矩阵做奇异值分解 \\
HX = U \Sigma V^T,有U^TU =E,V^TV = VV^T = E \\
\begin{aligned}
S & = \frac{1}{N} X^THX \\
 & = \frac{1}{N} (HX)^T (HX) \\
 & = \frac{1}{N} V \Sigma U^T U \Sigma V^T \\
 & = \frac{1}{N} V \Sigma^2V^T
\end{aligned}
$$
 所以说对方差特征值分解和对数据矩阵奇异值分解都可以进行计算。

 整和一下投影过程，我们可以将其转化为线性变换的形式，求得投影之后每个点的坐标(比较明显)
$$
\hat{HX} = HXV
$$
考虑矩阵$T = HX X^TH$
$$
T = HX(HX)^T = U \Sigma V^T V \Sigma U^T = U \Sigma^2 U^T
$$
PCA变换之后的坐标
$$
HXV =U \Sigma V^TV = U \Sigma
$$
因此当我们直接对矩阵$T$做特征值分解的时候，我们可以直接得到变换之后的坐标(结构)。

PCA可以对$S_{(d,d)}$做特征值分解，也可以对$T_{(N,N)}$做特征值分解，两者的区别是矩阵维度不同，因此，当遇到维度比较高的数据时，对$T$进行分解可以节省资源。





 