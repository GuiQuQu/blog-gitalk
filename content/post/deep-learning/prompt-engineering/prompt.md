---
title: "Prompt"
description: Prompt工程
date: 2023-05-23T15:24:59+08:00
image:
url: /deep-learning/prompt
math: true
comments: false
draft: false
categories:
    - Deep-Learning
---


# Prompt Engineering

## Zero-Shot

zero-shot是最简单的prompt的形式,将任务文本直接丢给LLM,让模型给出答案

例如如下情感分类任务
```
Text: I'll bet the video game is a lot more than a film.
Sentiment:
```

## Few-Shot

few-shot被zero-shot复杂一些,在给LLM任务文本，询问答案时，首先给出了任务相同的示例问题文本和答案，这样模型就能更好的理解人类的意图，往往可以给出更好的答案。

```
Text: (lawrence bounces) all over the stage, dancing, running, sweating, mopping his face and generally displaying the wacky talent that brought him fame in the first place.
Sentiment: positive

Text: despite all evidence to the contrary, this clunker has somehow managed to pose as an actual feature movie, the kind that charges full admission and gets hyped on tv and purports to amuse small children and ostensible adults.
Sentiment: negative

Text: for the first time in years, de niro digs deep emotionally, perhaps because he's been stirred by the powerful work of his co-stars.
Sentiment: positive

Text: i'll bet the video game is a lot more fun than the film.
Sentiment:
```

Few-shot中例子构建完全时经验性的,当改变**prompt的格式**,**例子的内容**或者是**例子的顺序**时,得到性能会显著的不同
[Zhao等人](https://arxiv.org/abs/2102.09690)基于GPT-3发现了LLM对于Few-Shot存在一些偏置,主要有以下3个
- Majority label bias,如果给定的例子不均衡,则会存在多数标签偏置
- Recency bias, 模型有生成他最后看到的例子的标签的倾向
- Common token bias, LLM倾向于生成更常见的token,因此一些稀少的token就很难出现。

Zhao等人在文章中提出了这三个问题,给出的方式时利用`N/A`这种输入来校准模型的输入

### 例子选择
- 利用kNN方法选择和给定任务文本语义最相似的例子([Liu等人,2021](https://arxiv.org/abs/2101.06804))
- 利用图的方法选择多样且具有代表性例子([Su等人,2022](https://arxiv.org/abs/2209.01975))
- 利用强化学习的方法来做采样选择([RL introduction](https://lilianweng.github.io/posts/2018-02-19-rl-overview/#q-learning-off-policy-td-control))

### 例子顺序的调整

empty empty empty 

## Instruction Prompting

Few-shot除了上述提到的不稳定因素外,Few-shot的最大问题还是占用了太多的token数量,因此可以输入的文本长度就会变短。

Instruct LM(e.g. [InstructGPT](https://openai.com/research/instruction-following))使用高质量的tuple(task instruction,input,ground truth output)来微调预训练模型
来使预训练模型可以更好的理解用户指令并且follow instruction。RLHF是一种常见的方式,使用instruct微调模型的好处是让模型和人类倾向更加对齐,从而减少了沟通的代价

**使用Instruction**

(1)当和instruct LM进行交互时,应该尽可能详细且准确和表示任务需求的细节,要详细的规定要做什么。

```
Please label the sentiment towards the movie of the given movie review. The sentiment label should be "positive" or "negative". 
Text: i'll bet the video game is a lot more fun than the film. 
Sentiment:
```
(2) 想模型说明目标观众
- 例如,为孩子生成教育材料
    ```
    Describe what is quantum physics to a 6-year-old.
    ```
- 生成安全内容
   ```
   ... in language that is safe for work.
   ```

**In-context instruction learning**([Ye等人,2023](https://arxiv.org/abs/2302.14691))将few-shot和instruction prompt结合,简单来说,就是few-shot给的例子是instruction的形式

## Self-Consistency Sampling
**Self-Consistency Sampling**([Wang等人](https://arxiv.org/abs/2203.11171))是一种多次采样temperature > 0的输入然后从候选输出中选择最好的输出的方法。不同任务好的标准不同,一个通常的解决方案是选择得票数最多的那个。对于已于验证的任务,例如带有单元测试的编程问题,我们可以简单使用单元测试来验证正确性

## Chain-of-Thought(CoT)

**Chain-of-Thought(CoT) prompting**([Wei et al. 2022](https://arxiv.org/abs/2201.11903))生成一系列的短句子来一步一步描述推理的逻辑(named as *resoning chains or rationales*),然后最终生成答案。CoT对于复杂推理任务的好处是更加显而易见的。当使用大模型(param > 50B)时,简单任务也能从CoT中轻微受益。

### Type of CoT prompts

主要有两种
- **Few-shot CoT** 在Few-shot给的例子中,每一个例子均包含了手写的或者是模型生成的推理链

(数学问题数据集)

```
Question: Tom and Elizabeth have a competition to climb a hill. Elizabeth takes 30 minutes to climb the hill. Tom takes four times as long as Elizabeth does to climb the hill. How many hours does it take Tom to climb up the hill?
Answer: It takes Tom 30*4 = <<30*4=120>>120 minutes to climb the hill.
It takes Tom 120/60 = <<120/60=2>>2 hours to climb the hill.
So the answer is 2.
===
Question: Jack is a soccer player. He needs to buy two pairs of socks and a pair of soccer shoes. Each pair of socks cost $9.50, and the shoes cost $92. Jack has $40. How much more money does Jack need?
Answer: The total cost of two pairs of socks is $9.50 x 2 = $<<9.5*2=19>>19.
The total cost of the socks and the shoes is $19 + $92 = $<<19+92=111>>111.
Jack need $111 - $40 = $<<111-40=71>>71 more.
So the answer is 71.
===
Question: Marty has 100 centimeters of ribbon that he must cut into 4 equal parts. Each of the cut parts must be divided into 5 equal parts. How long will each final cut be?
Answer:
```

-  **Zero-shot CoT** 
  使用一些自然语句来鼓励模型多步思考,例如使用`Let's think step by step`来鼓励模型首先生成推理链,然后用`Therefore,the answer is`来生成答案(两个阶段)([Kojima et.al 2022](https://arxiv.org/abs/2205.11916))
  或者是类似的语句`Let's work this out a step by step to be sure we have the right answer`([Zhou et al.2022](https://arxiv.org/abs/2211.01910))

# Automatic Prompt Design

Prompt是一个前缀token的序列,用来提高得到我们想要的输入的可能性。因此我们也可以将其作为训练参数直接优化。
相关的工作：
**AutoPrompt**([Shin et al.,2022](https://arxiv.org/abs/2010.15980))
**Prefix-Tuning**([Li & Liang,2021](https://arxiv.org/abs/2101.00190))
**P-tuning**([Liu et al.2021](https://arxiv.org/abs/2103.10385))
**Prompt-Tuning**([Lester et al.2021](https://arxiv.org/abs/2104.08691)),
这些方法的趋势是设置不断简化

**APE**(Automatic Prompt Engineer,[Zhou et al.2022](https://arxiv.org/abs/2211.01910))是一种自动选择prompt的方法，在模型生成的instruction候选池中搜索，然后根据一个选定的得分函数来过滤，最终选择得分最好的最好候选结果

自动构建CoT prompt
([Shum et al. 2023](https://arxiv.org/abs/2302.12822))

采用聚类的技巧构建CoT prompt
([Zhang et al. 2022](https://arxiv.org/abs/2210.03493))

# Augmented Language Models

A survey on augmented language model by [(Mialon et.al.2023)](https://arxiv.org/abs/2302.07842)


推荐阅读[controllable text generation](https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/#rl-fine-tuning-with-human-preferences)

Reference

[Lil' Log Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)