---
title: "torch 分布式训练"
description:
date: 2022-04-20T17:24:55+08:00
url: /deep-learning/torch-ddp
math: false
draft:  false
categories:
    - deep-learning
---

# torch分布式数据并行训练

主要使用的包是以下两个

```python
import torch.distributed as dist
import torch.multiprocessing as mp
```

我们主要关注**单机多卡**训练。

# 基本原理

`torch.distributed`的并行方法采用数据并行的方式，在多个gpu上加载相同的模型，然后给不同的gpu分发不同的数据，在计算完成之后汇总起来，完成反向传播，之后统一各个gpu上模型的参数，这样就完成了一次迭代。

# `torch.nn.DataParallel`和`torch.distributed.DistributedDataParallel`的比较

`torch.nn.DataParallel`简称为`dp`

`torch.distributed.DistributedDataParallel`简称为`ddp`

`dp`虽然在代码修改上很简单，但是`dp`的效率没有`ddp`好。`dp`在实际运行是单进程多线程的，但是因为python有`GIL`锁的存在，所以并没有完全并行起来。而在在使用`dp`的过程中，会存在显存分配不均衡的问题，即一张卡占用很多，另一张卡占用很少。

而`ddp`结合`multiprocess`使用，实际运行是多进程的，并行的效率更高，而且`ddp`的显存占用均衡的多

在[torch的官方文档](https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead)中，也推荐使用`ddp+mp`代替`dp`做单机多卡训练。

# 分布式训练基本概念

- `group` ：进程组，
- `world_size`： 全局的进程数量
- `rank`：表示进程序号，一般作为区分进程的唯一标识符使用
- `local_rank`: 进程内的GPU编号，非显式参数，由 `torch.distributed.launch` 内部指定

# `torch.multiprocessing`

`torch`为我们提供了`torch.distributed.launch`命令行工具来启动分布式训练，但是我仍然推荐自己写代码来手动提交多进程启动并行，这样可修改性更高。

`torch.multiprocessing`和python原生提供的`multiprocessing`兼容，但是原来的多进程不支持`cuda`操作，因此想要在`gpu`上多进程训练，需要使用`torch.multiprocessing`

## Step 1.指定开始方法

参看[torch的官方文档](https://pytorch.org/docs/stable/notes/multiprocessing.html?highlight=set_start_method)

```python
torch.multiprocessing.set_start_method(str)
```

可选的开始方法有`fork`,`spawn`,和`forkserver`,CUDA运行时不支持`fork`,需要使用`spawn`或者`forkserver`,我目前只用过`spawn`,因此下面针对`spawn`介绍，这里设置启动方法为`spawn`

## Step 2.提交多进程启动程序

```python
import torch.multiprocessing as mp
mp.spawn(fn, args=(), nprocs=1, join=True, daemon=False) # 启动函数
```

参数解释

- `fn` 进程的启动函数，一般我们定义`main_worker`函数传入，主要该函数的形式为`fn(i,*args)`,`i`是进程的编号,`*args`是传入的参数，因此我们也需要实现`fn(gpu,args...)`,这里使用gpu的代号只进程的编号
- `args`传入的参数
- `nprocs` 启动的进程数量，在单机多卡训练中，我们是针对每一个gpu分配一个进程的，因此在单机多卡我们我们设置`nprocs`为使用的`gpu`的数量。
- `join`  在所有进程间执行阻塞式`join`，这里我不太懂，但是我使用的一份代码默认该参数即可运行，不需要修改
- `daemon`设置为`True`时将会创建守护进程,守护进程时指脱离终端可以在后台运行的进程，一般我们用不到

参数总结

首先我们需要传入主程序入口`fn(i,*args)`和参数`args`,其次我们需要指定启动的进程数量`nprocs`，其他内容默认即可。

<span style="color: lightblue; font-size: 24px; font-weight: bold;">Note</span>

```
	1.为了保证一个进程只使用一张卡，我们需要在main_worker里指定当前进程使用的gpu设备,可以采用如下代码设置
	torch.cuda.set_device(gpu_id)
	2.该方法提交多进程时，当一个进程以非0状态退出时，其他进程也会被直接kill掉，我们可以少担心一些内存泄漏问题
```



# `torch.distributed`

## Step 1.初始化默认进程组和`distributed`包

为了使用`distributed`包，我们需要使用`torch.distributed.init_process_group`初始化默认进程组

```python
torch.distributed.init_process_group(backend, 
									 init_method=None, 
									 timeout=datetime.timedelta(seconds=1800), 
                                     world_size=- 1, 
                                     rank=-1, 
                                     store=None, 
                                     group_name='', 
                                     pg_options=None)
```

参数说明

- `backend` 指定分布式的通信后端，可选字符串有`mpi`,`gloo`,`nccl`,使用`gpu`推荐`nccl`,使用`cpu`填`gloo`,效果更好

- `init_method`该参数是多进程通信的方式，有以下两种，单机基本上走过场，我们照着写好就可以了。

  **使用tcp初始化**

  ```python
  import torch.distributed as dist
  
  dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456',
                          rank=rank, world_size=world_size)
  ```

  tcp初始化的格式为`tcp://ip地址:端口`，这里ip地址要填`master`进程所在主机的ip地址，单机即为本机，后面找一个不被占用的端口即可

  **使用共享文件系统初始化**

  ```python
  import torch.distributed as dist
  
  dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile',
                          rank=rank, world_size=world_size)
  ```

  没用用过，看到参考文档说这种方式容易出问题，这里更**推荐使用tcp初始化**，简单高效，不出问题。

  **环境变量初始化**

  如果不指定`init_method`,默认值为`	env://`,从环境变量中读取

  ```
  MASTER_PORT: master(rank0)机器上的空闲端口
  MASTER_IP: master(rank0)机器的ip
  WORLD_SIZE: 和参数中的world_size,也可以在这里指定
  RANK: 本机(当前进程)的RANk,和参数的rank同含义
  ```

  ​						**仍然推荐采用第一种方式初始化**

- `world_size` 全局总共的进程数量，在单机多卡中即为gpu的数量

- `rank`当前进程的`rank`标识符，在``分布式基本概念``那里已经介绍过了

- `sotre`和`init_method`冲突，我们只需要指定一个即可。torch文档里面说是所有进程都可以访问的key/value存储，用于交换连接和地址信息。

## Step 2. 包装模型

```
torch.nn.parallel.DistributedDataParallel(
    module, 
    device_ids=None, 
    output_device=None, 
    dim=0, 
    broadcast_buffers=True, 
    process_group=None, 
    bucket_cap_mb=25, 
    find_unused_parameters=False, 
    check_reduction=False, 
    gradient_as_bucket_view=False, 
    static_graph=False)
```

介绍一些常用的参数

- `model`,传入需要包装的模型，需要主要我们需要现在模型放到gpu上

- `device_ids`  使用的gpu_id列表，在单机多卡上，我们只能传入一个设备id，对于多机和cpu模型，我们只能设为`None`

- `output_device`  模型输出结果存放的设备id，默认值为`device_ids[0]`，一般的默认值都是device_ids[0]

- `find_unused_parameter`,遍历一遍`autograd graph`，寻找不需要求导或者`forward`函数没有使用的参数，将其直接推进到`reduce`阶段。我们可以设置为`True`

  下面是一行使用的示例代码(单机多卡)

  ```python
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
  ```

`ddp`的简单原理

​	`ddp`在各进程梯度计算完成之后，各个进程需要将梯度进行汇总平均(`all-reduce`)到rank0的进程中，然后再广播到其他进程中，其他进程利用该梯度独立更新参数，因此再最一开始各个进程的初始参数就一致（初始时进行一次广播），因此采用这种方式可以保证每个进程的模型都是一模一样的。

## Step 3. 为数据集创建DistributedSampler

```python
if torch.distributed.is_initialized():
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None # args.distributed 指定是否采用分布式训练
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle, # 只有再非分分布式下才有用
            num_workers=0,
            pin_memory=True,
            sampler=sampler,
            drop_last=is_train,
        )
```

和普通的创建`dataloader`的过程几乎没有差别

这里需要说明以下几点

- `shuffle`和`sampler`冲突，当我们指定了`sampler`之后`shuffle`就没用了

- 这里`batch_size`是指单卡的`batch_size`,当我们使用多卡的时候，我们实际上`total_batch_size = world_size * batch_size`

- `torch.distributed.is_initialized()`我们可以检查每个进程是否都完成了初始化

- `DistributedSampler`在最开始就会将数据完全切开，这样每一个进程只能看到一部分数据，各个进程之间独立，不会发生重叠。采样器shuffle默认为`True`。但是这种打乱只能在当前进程可以看到的一小部分数据上进行，无法充分打乱数据。

  这样做的好处在多机多卡上可以避免数据在不同机器上的传输，但是在单机多卡上会给我们带来新的问题

  在torch的官方文档下，有以下<span style="color: red; font-weight: bold">警告</span>

  ```
  In distributed mode, calling the set_epoch() method at the beginning of each epoch before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used.
  ```

  因为**sampler只能看到小部分数据**，因此在每一个epoch创建dataloader的迭代器之间，我们需要调用`set_epoch()`方法充分打乱数据，否则**每一个epoch的数据顺序是完全一样的。**

  示例代码(来自[torch官方文档](https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.distributed.DistributedSampler))

  ```
  sampler = DistributedSampler(dataset) if is_distributed else None
  >>> loader = DataLoader(dataset, shuffle=(sampler is None),
  ...                     sampler=sampler)
  >>> for epoch in range(start_epoch, n_epochs):
  ...     if is_distributed:
  ...         sampler.set_epoch(epoch)
  ...     train(loader)
  ```

- 评测问题，因为train_data和dev_data都采用了`DistributedSampler`,因此每一个进程也只能看到一部分的评测数据，如果我们仅指定master进程做评测的话，我们无法使用全部的评测数据，但是还是能看一个大致的效果。我们也可以使用下文提到的`all_gather`函数把模型的所有输出都汇聚起来，然后再做评测。

## Step 4. 训练模型

和单机训练区别不大，我会把我已知的区别列出来

- 采用ddp或者dp之后，模型会被一层`module`类包装，因此需要使用`m = model.module`才可以找到你原来的模型，去寻找你设置的类属性和方法，但是一般`forward`方法会继承下来。因此我们采用分布式训练时，保存得到的状态词典的key都会带`module.`的前缀

- 有一些分布式的api可以帮助我们汇总各个进程的计算结果，在计算损失时我们可能会用到。

  举例 CLIP模型计算损失希望batch越大越好，因此我们可以把每个进程计算得到特征采用`all_gather`汇聚起来计算交叉熵损失

  可以结合[官方文档](https://pytorch.org/docs/stable/distributed.html?highlight=all_gather#torch.distributed.all_gather)使用，而且一般的汇聚操作，如果我们不指定进程组，都是在默认进程组下做，即我们使用`init_process_group`初始化的进程组

<span style="color: lightblue; font-size: 24px; font-weight: bold;">Note</span>

- 模型的保存，在保存中上文已经说过状态词典的key值会加`modules.`前缀
- 模型的加载，1）在单机单卡模型做推理时，去掉`modules.`前缀  2 ）当为不同的进程加载模型时，我们需要把模型放到对应的该进程使用的设备上，具体地，我们需要添加`map_location`参数，`torch.load(state_dict.pt,map_location=f"cuda:{gpu}")`

# logging

由于开了多进程，如果使用普通的logger会让进程之间的输出混乱，因此我们需要带有消息队列的logger，可以直接复制以下代码到`logger.py`里面去

使用时步骤

**Step 1.** 

```python
log_queue = setup_primary_logging(log_file, level)
```

`log_queue`等待被传入各个进程中，因此函数`main_worker`需要把`log_queue`传入

**Step 2.**

在各个进程中调用

```python
setup_worker_logging(rank, log_queue, level)
```

随后就可以使用logging.info进行输出了，并且会根据rank区分是哪一个进程的输出。

一般我们在打印损失的时候只需要打印master进程的结果就可以了。但是如果不做各个进程之间的汇聚，这个loss只是rank0进程计算得到的结果，我们可以利用`all_gather`得到每一个进程的loss，然后求平均让rank0的进程输出。

```python
import argparse
import logging
from logging import Filter
from logging.handlers import QueueHandler, QueueListener

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Queue

# 初始化logger的消息队列
def setup_primary_logging(log_file, level):
    log_queue = Queue(-1)

    file_handler = logging.FileHandler(filename=log_file)
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', 
        datefmt='%Y-%m-%d,%H:%M:%S')

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    file_handler.setLevel(level)
    stream_handler.setLevel(level)

    listener = QueueListener(log_queue, file_handler, stream_handler)

    listener.start()

    return log_queue

# 根据rank区分不同进程的输出
class WorkerLogFilter(Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f"Rank {self._rank} | {record.msg}"
        return True

# 在某个进程中声明logger,该进程的rank被传入
def setup_worker_logging(rank, log_queue, level):
    queue_handler = QueueHandler(log_queue)

    worker_filter = WorkerLogFilter(rank)
    queue_handler.addFilter(worker_filter)

    queue_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.addHandler(queue_handler)

    root_logger.setLevel(level)


def fake_worker(rank: int, world_size: int, log_queue: Queue):
    setup_worker_logging(rank, log_queue, logging.DEBUG)
    logging.info("Test worker log")
    logging.error("Test worker error log")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:6100',
        world_size=world_size,
        rank=rank,
    )

if __name__ == "__main__":
    # Set multiprocessing type to spawn
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu-list", type=int, help="List of GPU IDs", nargs="+", required=True)

    args = parser.parse_args()

    world_size = len(args.gpu_list)

    # Initialize the primary logging handlers. Use the returned `log_queue`
    # to which the worker processes would use to push their messages
    log_queue = setup_primary_logging("/usr/lusers/gamaga/out.log", logging.DEBUG)

    if world_size == 1:
        worker(0, world_size, log_queue)
    else:
        mp.spawn(fake_worker, args=(world_size, log_queue), nprocs=world_size)
```

# <span style="color: red; font-weight: bold">Warning</span>

写好分布式代码之后直接使用命令行运行程序，避免使用任何IDE，否则我们还需要看IDE的运行实现，非常麻烦。

# 主要参考内容

1. [PyTorch 多进程分布式训练实战](https://murphypei.github.io/blog/2020/09/pytorch-distributed)

2. [pytorch多gpu并行训练](https://zhuanlan.zhihu.com/p/86441879)

3. [torch官方文档](https://pytorch.org/docs/stable/index.html)