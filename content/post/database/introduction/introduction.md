---
title: "第一章 介绍"
description: 
date: 2023-10-07T23:38:22+08:00
image:
url: /db/introduction
math: false
comments: true
draft: false
categories:

tags:
    - Database
---

# 介绍

什么是数据(Data)？
- 数据:能够被记录且具有实际含义的已知事实

什么是数据管理(Data Management)?
- 数据管理:在计算机种对数据进行存储,检索,更新,共享

什么是数据库?

- A database(DB) is a set of related data that is organized(组织), shared(共享) and persistent(持久化)

数据库管理系统(database management system,DBMS)

- a database management system(DBMS) is a general-purpose system software that facilates the organization(组织), storage(存储), manipulation(操作), control(控制), and maintainence(维护) of databases among various users and applications.

数据库用户
- 数据库管理员(database administrator,DBA)
- 数据库设计者(database designer)
- 终端用户(end user)

数据库模式(Database Schema)

数据库模式(database schema)是对数据库的结构,类型,约束的描述
- 数据库模式是数据库的"类型声明"
- 数据库模式不经常变化

例如 Student的关系模式(表头)

| Sno | Sname | Ssex | Sage | Sdept |

数据库实例(Database Instance)(表的内容)

数据库示例是数据库在某一特定时间存储的数据
- 数据库示例是数据库的"值"
- 每当数据库被更新,数据库实例就会发生变化

数据库语言(Database Language)

数据库语言是用户/应用程序与DBMS交互时所使用的语言
- 数据定义语言(data definition language,DDL),DBA和数据库设计者使用用来声明数据库模式的语言
- 数据操纵语言(data manipulation language,DML),查询和更新数据库时所使用的语言