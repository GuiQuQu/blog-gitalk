---
title: "EMA"
description:
date: 2023-04-20T17:24:55+08:00
url: /deep-learning/ema
math: true
draft:  false
categories:
    - Deep Learning
---

[Reference](https://zhuanlan.zhihu.com/p/68748778)

指数移动平均（Exponential Moving Average）也叫权重移动平均（Weighted Moving Average），是一种给予近期数据更高权重的平均方法。

假设我们有n个数据$[\theta_1,\theta_2,...,\theta_n]$

- 普通的平均数 $\overline v = \frac{1}{n} \sum_{i=1}^n \theta_i$

- EMA算法$v_t = \beta v_{t-1} + (1- \beta) \theta_t$,$v_t$代表前t条数据的结果($v_0 =0$),$\beta$是权重值，一般设为0.9~0.999



在深度学习中的优化过程中，$\theta_t$是模型t时刻的模型权重，$v_t$是t时刻的影子权重，在梯度下降过程中，会一直维护这个影子权重，但是这个影子权重并不会参与训练。

pytorch实现

```python
import torch.nn as nn

class EMA():
    def __init__(self,model:nn.Module,decay) -> None:
        self.model = model
        self.decay = decay # beta
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# init
ema =EMA(model,decay=0.999)
ema.register()

# 训练过程中,更新完参数后,同步update shadow weights
def train():
    optimizer.step()
    ema.update()

# 在eval前,修改模型参数为shadow weight,eval之后,在恢复原来模型的参数继续训练
def evaluate():
    ema.apply_shadow()
    # evaluate
    ema.restore()
```





