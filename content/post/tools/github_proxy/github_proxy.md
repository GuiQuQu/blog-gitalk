---
title: "使用代理访问GitHub"
description: 
date: 2023-05-25T17:36:49+08:00
image:
url: /github_proxy
math: false
comments: false
draft: false
categories:
    - Tools
---

这篇文章的主要目的是使用代理访问github的方式说清楚

# 应用场景

在有了代理之后,经常会遇到一种情况,使用浏览器等工具访问github完全正常,但是往往`git clone`的时候总是clone不下来,
然后我们就会退了求其次,在github的仓库自己手动把源码下载下来。

但是在面对一个使用git的工具的时候,仓库的数量不过来,并且我们也并不清楚对应仓库版本的哈希值,导致下载仓库非常棘手,日常超时

一个典型的场景是pip下载python包时,有的包往往是来自于git仓库的

- pip下载git仓库

下面就来说明如何使用代理来访问git仓库

**前提:你已经拥有了一个代理**

一般而言,我们的代理都是在本机的某一个端口
例如,使用clash,最常见的两种代理协议http(7890)和socks5(7891),分别对应的url是

```shell
http://127.0.0.1:7890 
```
```shell
socks5://127.0.0.1:7891
```

拉取和提交git仓库基于两种协议：http和ssh,下面分别说明

# 设置http(s)代理

设置git的http(s)代理的方式如下

## 针对所有域名设置代理(不推荐)

例如从huggingface的仓库中下载模型也是用git,不过远程仓库的域名就不是github.com了

> 使用 http代理

```shell
git config --global http.proxy http://127.0.0.1:7890 
git config --global https.proxy http://127.0.0.1:7890
```

> 使用socks5代理 

```bash
git config --global http.proxy socks5://127.0.0.1:7891
git config --global https.proxy socks5://127.0.0.1:7891
```

## 只针对GitHub设置代理(推荐)

> 使用 http代理

```shell
git config --global http.https://github.com.proxy http://127.0.0.1:7890 
git config --global https.https://github.com.proxy http://127.0.0.1:7890
```

> 使用socks5代理 

```bash
git config --global http.https://github.com.proxy socks5://127.0.0.1:7891
git config --global https.https://github.com.proxy socks5://127.0.0.1:7891
```

## 取消代理

> 取消全域名代理

```shell
git config --global --unset http.proxy
git config --global --unset https.proxy
```

> 取消对应域名代理

```shell
git config --global --unset http.https://github.com.proxy
git config --global --unset https.https://github.com.proxy
```

# 设置ssh代理

https代理做身份验证,意味着每次交互都必须输入账号和密码,但是对应clone公共仓库来说已经够用了。

但是如果我们自己需要push代码上去,使用ssh来连接仓库体验更好,因为不需要一直输入账号和密码

使用ssh代理可以完全免去输入用户和密码

前提
- 保证自己的电脑上已经设置了ssh_key,并且和github做绑定(使用`ssh-keygen`命令创建,一路enter)

```shell
# Linux,MacOS
vim ~/.ssh/config
# windows
create file named 'config' in C:\Users\your_user_name\.ssh
```
将下面的内容添加在config文件中
对于windows用户来说,需要使用connect.exe,下载Git都会带
例如我的位置是`C:\Program Files\Git\mingw64\bin\connect.exe`
在linux或者MacOS下会使用`nc`完成连接
(在Linux下,`nc`,可以使用`type nc`来查看这个可执行程序的具体位置)

```shell
# windows
Host github.com
    Port 22
    User git
    HostName github.com
    IdentityFile "C:\Users\your_user_name\.ssh\id_rsa"
    ProxyCommand "C:\Program Files\Git\mingw64\bin\connect" -S 127.0.0.1:7891 -a none %h %p
    TCPKeepAlive yes

Host ssh.github.com
    Port 443
    User git
    HostName ssh.github.com
    IdentityFile "C:\Users\your_user_name\.ssh\id_rsa"
    ProxyCommand "C:\Program Files\Git\mingw64\bin\connect" -S 127.0.0.1:7891 -a none %h %p
    TCPKeepAlive yes


# linux or MacOS
Host github.com
    Port 22
    User git
    HostName github.com
    IdentityFile "\home\user_name\.ssh\id_rsa"
    ProxyCommand nc -v -x 127.0.0.1:7891 %h %p
    TCPKeepAlive yes

Host ssh.github.com
    Port 443
    User git
    HostName ssh.github.com
    IdentityFile "\home\user_name\.ssh\id_rsa"
    ProxyCommand nc -v -x 127.0.0.1:7891 %h %p
    TCPKeepAlive yes
```

保存完成之后,使用下面的命令测试到`github.com`的ssh连接

```shell
ssh -T git@github.com
```

# 参考

[设置代理解决github被墙](设置代理解决github被墙)

[一文让你了解如何为 Git 设置代理](https://ericclose.github.io/git-proxy-config.html) 


