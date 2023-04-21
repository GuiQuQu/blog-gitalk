---
title: "CLIP 源码分析"
description:
date: 2022-04-20T17:24:55+08:00
url: /deep-learning/clip
math: true
draft:  false
categories:
    - Deep Learning
---

CLIP结构

# 视觉

- ModifiedResNet
- VisionTransformer

## ModifiedResNet

### 2维卷积操作

输入的大小$(N,C_{in},H,W)$,输出的大小$(N,C_{out},H_{out},W_{out})$

需要设定的参数

- `in_channels`,输入图像的通道数,$C_{in}$
- `out_channels`,输出图像的通道数，$C_{out}$,也决定了实际卷积核的个数
- `kernel_size`,卷积核高宽，实际卷积核大小为`(C_in,kernel_h,kernel_w)`
- `stride=1`,步距
- `padding=0`,在图像周围padding的大小
- `dalation=1`
- `groups=1`
- `bias=True`，是否加上标量偏差
- `padding_mode=zeros`,指定padding的模式，一般都是补零，不需要动

卷积核也叫做`filter`，因为在时域的卷积操作是和频域的乘积等价的，在频域我们可以通过滤波器来去除特定频率的信号，从而得到和在时域做卷积相同的效果。

判别CNN的卷积核个数和卷积核大小

我们可以先看pytorch官网给定的计算公式
$$
out(N_i,C_{out_j}) = bias(C_{out_j}) + \sum_{k=0}^{C_{in}-1} weight(C_{out_j},k) \star input(N_i,k)
$$
其中，$\star$是互相关操作，我们得到第i张图像的第j个通道的输出结果时，bias是标量偏差，在互相关操作中，我们需要在输入图像的每一个通道上分为使用该通道的卷积核$weight(C_{out_j},k)$和输入图像对应元素做互相关操作，并且我们会将输入图像在多个通道上计算的结果加起来。

通俗的说法，就是在多通道情况下，我们的卷积核也是作用在多个通道的。比如说输入图像大小为`(3,224,224)`,我们设定卷积核大小为`(3,3)`，但是实际上我们每一个输出通道的卷积核大小为`(in_channels,3,3)`

判断CNN的输出大小

在考虑了`stride`和`padding`的情况下，CNN的输出大小为

$$
(N,C_{out},\lfloor \frac{h_{in} + 2 * padding - h_{kernel}}{stride_h}\rfloor + 1，\lfloor\frac{w_{in} + 2 * padding - w_{kernel}}{stride_w}\rfloor + 1)
$$

这里给出一个比较形象的记法，padding很好理解，我们按照互相关的计算流程，第一次把卷积核叠在左上角，然后开始移动这个窗口，那么就横向移动来说，我们只可以移动横向方向上这个卷积核还没有盖上的位置，每次移动的步距都是stride,因此是$\frac{h_{in} + 2 * padding - h_{kernel}}{stride_h}$,并且当后面空间不够没法把卷积核完全叠上的之后就不做了，因此需要取下整。之后在加上我们最开始叠的那个位置，因此后面有个+1.

因为pooling操作只是在互相的计算流程改变了，因此pooling之后的大小也可以这么计算，注意pooling操作的stride默认值是kernel_size，因为做pooling的时候不重叠。

2维卷积操作的流程。

2维卷积操作使用名为互相关（cross-correlation）的操作来完成，互相关就是我们在学习卷积的时候进行的操作

```python
## 卷积操作
import torch
from torch import nn

input_image = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
kernel = torch.tensor([[0,1],[2,3]])
def corr2d(X,K):
    h,w = K.size()
    Y = torch.zeros((X.size()[0] - h + 1, X.size()[1] - w + 1))
    for i in range(Y.size()[0]):
        for j in range(Y.size()[1]):
            Y[i,j] = torch.mul(X[i : i + h, j : j + w],K).sum()
    return Y
print(corr2d(input_image,kernel))
# output
tensor([[19., 25.],
        [37., 43.]])
```

二维的卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出，卷积层的模型参数包括了卷积核核标量偏差。

[卷积判别图像边缘](https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter05_CNN/5.1_conv-layer)，从这个例子中，我们可以得到卷积操作可以帮助有效的表征局部空间。

### ModifiedResNet

待补充...

## VisionTransformer

**1.如何将图像转换为transformer的输入格式**

利用patch,比如224\*224大小的图像，我们选定patch为16\*16,我们可以通过卷积实现

```python
# width: 768
# patch_size: 16
# 这里的conv1我认为和bert的embedding作用类似
self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
x: # size = (N,3,224,224)
x = self.conv1(x) # (N,768,14,14) # 最后两维就是序列长度
x = x.reshape(x.size()[0],x.size()[1],-1) # (N,768,14*14)
x = x.permute(0,2,1) # (N,14*14,768)
```

2.添加`cls_embedding`和`positional_embedding`

```python
# init
self.cls_embedding = nn.Parammeter(torch.randn(width)) # (768)
self.position_embedding = nn.Parameter(torch.randn((224 // 16) ** 2) + 1,width) # (14*14 + 1, 768) cls_token + 每一个patch位置

# forward
# add cls_embedding
x = torch.cat([self.cls_embedding.to(x.dtype) + torch.zeros(x.size()[0],1,x.size()[-1],dtype=x.dtype,device=x.device),x],dim = 1) # (N 14*14+1,768)
# add position_embedding
x = x + self.positional_embedding.to(x.dtype) # (N,1+14*14,768) # (1+14*14,768) by boardcast
```

3.使用transformer计算

```python
x = self.ln_pre(x) # (N,1+14*14,768)
 x = x.permute(1, 0, 2) # (1+14*14,N,768) 需要变换维度的原因是这里用的是torch实现的nn.MultiheadAttention，默认不是batch_first的
 x = self.transformer(x)
 x = x.permute(1, 0, 2) # (N,1+14*14,768)
 x = self.ln_post(x[:, 0, :]) # cls embedding output
if self.proj is not None:
    x = torch.mm(x,self.proj) # (bs,hs) * (hs,embed_dim) = (bs,embed_dim)
return x
```

transformer内部计算过程(一层)

```python
x = x + self.attention(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
return x
```

- `ln_1`,`ln_2`，LayerNorm
- `mlp`:`Linear(hidden_size,hidden_size\*4),ACTFUN,Linear(hidden_size\*4,hidden_size)`
- `attention`:`self.attn(query,key,value,need_weights=False,attn_mask=self.attn_mask)[0]`
- `attn = nn.MultiheadAttention(hidden_size,n_head)`

记住self-attn公式就可以来算了
$$
output = softmax(\frac{Q^TK}{\sqrt{d_k}},dim = -1)V \\
Q\_size,K\_{size},V\_{size} = [N,seq\_{length},hidden\_{size}]
$$
一般会设定一个参数问是否输出`attention_score`,即$softmax(\frac{Q^TK}{\sqrt{d_k}},dim = -1)$

# 文本

- BERT(roberta-ext-wmm)

源码解析看[BERT解析](/deep-learning/bert)

