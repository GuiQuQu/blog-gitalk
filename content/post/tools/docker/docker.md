---
title: "Docker安装,使用,手册"
description: 
date: 2023-12-27T17:14:53+08:00
image:
url:
math: true
comments: true
draft: false
categories:
---

[Docker-Guide](https://docs.docker.com/get-started/overview/)
[Docker-commandline reference](https://docs.docker.com/reference/)
# Docker 安装

因为我使用的Docker是在安装ubuntu系统时自动安装的(使用snap安装的),因此这部分先跳过

# Docker 权限问题

当输入Docker命令时,如果我们不是sudo用户,我们会发现我们无法使用docker命令,并且显示我们没有权限,这个问题可以通过将我们自己的用户加入到docker用户组中来解决

问题出在`/var/run/docker.sock`的权限上,这个文件时root可读可写,docker用户组可读可执行的,我们不在这个范围内
(如果机器本身没有docker用户组,那么可能会变成root用户组)

```shell
cat /etc/passwd # 可以查看所有用户的列表

w  # 可以查看当前活跃的用户列表

cat /etc/group  # 查看用户组
```

```shell
sudo groupadd docker # 创建 docker 组
sudo usermod -aG docker $USER # 将用户加入到docker用户组
```

如果说你在改完了之后你发现`/var/run/docker.sock`的权限还是不对,那么可以尝试重启一下docker服务,或者干脆重启一下机器

```shell
sudo systemctl restart docker(snap.docker.dockerd)
```

[点击这里,参考docker官方网站的说明](https://docs.docker.com/engine/install/linux-postinstall/)

# Dockerfile 书写以及构建

Dockerfile是用来构建docker镜像的文件,我们可以通过书写Dockerfile文件来构建我们自己的docker镜像

[dockerfile指令参考](https://docs.docker.com/develop/develop-images/instructions/)

`FROM baseImage` 指定基础镜像,这个镜像是我们构建镜像的基础,我们可以在这个镜像的基础上进行修改,要求所有的Dockerfile文件都必须由这个指令开头

`RUN command`

[RUN指令参考](https://docs.docker.com/engine/reference/builder/#run)

RUN指令执行一次就会叠加一个额外的container layer,因此为缩小镜像的大小,我们需要将多个命令使用`&&`连接起来,这样就可以在一个container layer中完成多个命令的执行

- `CMD ["executable", "param1", "param2"]` (exec form) 
- `CMD ["param1", "param2"]` (exec form,作为 ENTRYPOINT 的默认参数)
- `CMD command param1 param2` (shell form)
[CMD指令参考](https://docs.docker.com/engine/reference/builder/#cmd)

CMD制定了这个这个镜像作为容器启动时执行的命令,一个Dockerfile只能有一个CMD命令,如果有多个,那么只有最后一个会生效

`ADD or COPY`

添加or复制文件

`ENTRYPOINT`

`ENTRYPOINT`的最佳方法是指定镜像的主命令,然后使用`CMD`来指定命令的选项

例如,`s3cmd`,这是一个命令行工具,我们想要执行`s3cmd --help`

```Dockerfile
ENTRYPOINT ["s3cmd"]
CMD ["--help"]
```

## docker build

[Reference](https://docs.docker.com/engine/reference/commandline/build/)

Usage

`docker build [OPTIONS] PATH | URL | -`

`docker build`命令用来从一个Dockerfile和一个"context"中来构建docker镜像

 一个build的上下文即使`PATH` or `URL` 中指定的文件集合,build的过程可以参考在context中的任何文件,例如,可以使用`COPY`指令来从上下文中复制文件到镜像中

使用`-`从标准输入中读取Dockerfile

```shell
docker build - < Dockerfile
```
如果你使用了`STDIN`or `URL`指向了一个文本文件,那么系统会把对应的文本内容作为`Dockerfile`,此时是没有上下文的,`-f, --file` 指定了`Dockerfile`文件路径选项失效

在多数情况下,最好的方法是把`Dockerfile`文件放在一个空目录下,然后在依次添加构建镜像需要的文件,不过docker也提供了`.dockerignore`文件来忽略一些不需要的文件

**Options**
- `-f, --file` 指定Dockerfile文件的路径
- `-t, --tag` 指定镜像的名字,tag可选,形式为`name:tag`

**Example**
```shell
# -tag 指定构建的镜像的名字
# . 使用的context是当前文件夹
docker build --tag=buildme .
```
重建镜像

在重建镜像之前,需要确保把构建缓存删除,确保我们是从头再次开始,使用以下的命令删除build的缓存
```shell
docker builder prune -af
```

如果需要将`docker build`的构建结果导出到本地文件系统,请参考[这里](https://docs.docker.com/build/guide/export/)

## Docker镜像相关的CLI命令
------------------
`docker image ls` or `docker images`, 列出镜像

Usage
```shell
docker image ls [OPTIONS] [REPOSITORY[:TAG]]
```
------------------

`docker image rm` or `docker rmi` 移除一个或者多个镜像,可以用镜像的名字,也可以用镜像的UID

Usage
```shell
docker image rm [OPTIONS] IMAGE [IMAGE...]
```
------------------
`docker image prune` 删除所有的没有被使用的镜像


Usage
```shell
docker image prune [OPTIONS]
```

------------------

## old content

**image**
- `docker pull ubuntu:22.04`, 从docker hub上拉取镜像
- `docker image ls` or `docker images`, 列出所有的docker镜像
- `docker rmi ubuntu:22.04` or `docker image rm ubuntu:22.04`, 删除名为`ubuntu:22.04`的镜像
- `docker [container] commit CONTAINER [CONTIANER_IMAGE_NAME[:TAG]]`, 保存容器为镜像
- `docker save -o ubuntu20.04`

**container**
- `docker run --name=buildme `, 开始运行名字为`buildme`的容器
- `docker exec -it buildme /bin/client`, 在`buildme`容器中运行`/bin/client`命令,并采用交互式的方式运行
- `docker stop buildme` 停止名为`buildme`的容器
- `docker ps `查看当前已经启动的容器, `docker ps -a` 查看所有的容器
- `docker attach buildme` 进入名为`buildme`的容器
   `Ctrl+p` + `Ctrl+q` 挂起容器, `Ctrl+d` 直接退出
- `docker rm buildme` 删除名为`buildme`的容器
- `docker container prune` 删除所有处于停止状态的容器
- `docker rename buildme buildme2` 将名为`buildme`的容器重命名为`buildme2`


# container相关

## docker run

[Reference](https://docs.docker.com/engine/reference/commandline/run/)

Usage:

`docker run [OPTIONS] IMAGE [COMMAND] [ARG...]`
docker run命令在容器内运行一个命令,如果需要的话,拉取并启动容器

**Option**
- `-d, --detach` 后台运行容器
- `-it`  `-i`交互式运行, `-t` 分配一个伪tty终端,在容器前台可以使用`Ctrl+p` + `Ctrl+q`
来挂起容器
- 一般都会同时使用`-itd`,表示交互式运行容器，后台运行容器
- `--name` 指定容器名字
- `--rm` 在容器停止后自动删除容器


例如
    
```shell
# 省略了[COMMAND],启动的容器名字为buildme2,
# --rm 在容器停止后自动删除容器
# --detach 后台运行容器
# --name 指定容器名字
# buildme是对应的镜像的名字
docker run --name=buildme2 --rm --detach buildme

# -it 表示交互式运行,可以使用Ctrl+p + Ctrl+q来挂起容器
# -t 分配一个伪tty终端
docker run --name=buildme --rm -it buildme
```

`-it`属性还是比较重要的,如果不给docker分配一个伪tty终端,而且给出的ENTRYPOINT不是死循环,那么这个docker容器在执行完这个命令就直接结束了，容器就直接停止了。

参考这个http-server的容器运行情况,我按下`Ctrl+C`就会直接结束,我在前台看到的也全是这个server这个二进制文件的输出日志
```shell
(base) klwang@itnlp80:~/code/docker-test/test1$ docker run --name=buildme --rm buildme
2023/12/29 09:57:00 Starting server...
2023/12/29 09:57:00 Listening on HTTP port 3000
```

## docker start
[Reference is here](https://docs.docker.com/engine/reference/commandline/start/)

Usage

`docker start [OPTIONS] CONTAINER [CONTAINER...]`

## docker attach
[Reference is here](https://docs.docker.com/engine/reference/commandline/attach/)

将一个docker容器的STDIN,STDOUT,STDERR连接到当前的终端,也就是进入容器

Usage

`docker attach [OPTIONS] CONTAINER`

## docker exec

在已经运行的容器内执行命令

Usage

`docker exec [OPTIONS] CONTAINER COMMAND [ARG...]`

## docker ps

列出容器

Usage

`docker ps [OPTIONS]`

**Options**
- `-a, --all` 列出所有的容器,包括已经停止的容器
