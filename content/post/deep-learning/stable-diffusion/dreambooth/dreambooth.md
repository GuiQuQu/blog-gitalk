---
title: "Dreambooth"
description: 
date: 2023-04-24T16:23:27+08:00
image:
url: /stable-diffusion/dreambooth
math: true
comments: false
draft: false
categories:
    - stable-diffusion
---

**DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation**



文本的主要目的是通过微调sd预训练模型，让模型可以做到可以生成特定对象的图像



> We fine-tune the text-to-image model with the input images and text prompts containing a unique identifier followed by the class name of the subject (e.g., "A [V] dog").

微调模型的输入：

- input images
- text prompt(包含一个特殊的标识符表明图像所属的类别)

这种方法仅仅只提供3到5张照片就可以训练，还需要额外提供的信息有

1. 一个特殊的标识符,模型会将这个特殊的标识符和图像中的主题绑定(这个标识符需要比较稀有，使得语言模型对于这个词具有较弱的先验)
2. 这3-5张图像所属的类(例如，提供5张某个小狗的图片，这类别就是狗)

如果我们只使用图像和一个特殊的标识符进行微调，会造成语言漂移(language drift)现象,语言漂移是只本来在大量数据上训练得到的预训练模型，然后在特定任务上微调会逐渐失去语言的语义和语法知识。

在sd中，仅仅使用特殊标识符和图像进行建模，会使得模型逐渐忘记本来粗类别的知识。

比如我们要求生成狗的图像，但是模型实际上只会生成这只特定的狗，其他的狗就不会被生成，模型已经认为狗就这一种样子

因此这种方式也造成了模型丢失了多样性。

为了克服上面两个问题，作者使用了模型自己的输出作为监督信号。

在训练时，模型需要生成新的图像数据$x_{pr} = \hat{x}(z_{t_1},c_{pr})$

$z_{t_1}$是标准高斯噪声，$c_{pr}$ 是条件先验，使用的条件是一个prompt：`a [class noun]`，比如狗就是`a class dog`,然后结合原来sd的损失得到新的损失
$$
\mathbb{E}_{x,c,\epsilon,\epsilon^{\prime},t}[w_t \parallel \hat{x}_\theta(\alpha_t x + \sigma_t \epsilon,c) -x\parallel^2_2 + \lambda w_t^{\prime}  \parallel \hat{x}_\theta(\alpha_t x_{pr}+ \sigma_t \epsilon,c) -x_{pr}\parallel^2_2]
$$
