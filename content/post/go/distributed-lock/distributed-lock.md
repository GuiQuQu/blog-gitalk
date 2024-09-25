---
title: "分布式锁"
description: 
date: 2024-03-28T13:25:51+08:00
image:
url:
math: true
comments: true
draft: false
categories:
    - Distributed System
---

利用Redis实现分布式锁

分布式锁是控制分布式系统之间同步访问共享资源的一种方式，通过互斥来保持一致性。

通过在Redis中利用`SETNX` 操作来设值,设值成功即获取锁成功,设值失败即获取锁失败。

例如 `SETNX key uuid EX sec` 这个操作是原子操作

对`key`上错,每一个进程上锁时都有自己的一个`uuid`,这样可以保证只有持有锁的进程才能释放锁。解锁时,先检查uuid是否一致,一致则删除key,释放锁。 (这个检查然后del也需要时原子操作,可以利用Lua脚本实现)

自动续租,就go来说,是在上锁之后开启一个额外的goroutine,定期刷新过期时间
