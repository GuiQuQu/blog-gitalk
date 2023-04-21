# !/bin/bash
# 该脚本用于每日自动更新博客内容
# 请使用root权限来执行脚本

blog_path="/root/guiququ-blog"
cd $blog_path
# 从git仓库拉取内容
git pull
if [ $? != "0" ]; then
    echo "git pull fail"
    exit 1
fi
# 执行博客处理脚本
python3 precess_md.py
if [ $? != "0" ]; then
    echo "precess_md.py fail"
    exit 1
fi
# 重新构建blog
$GOPATH/bin/hugo
if [ $? != "0" ]; then
    echo "hugo build fail"
    exit 1
fi
# 复制到 /var/www文件夹下

cp -r $blog_path/public/* /var/www
if [ $? != "0" ]; then
    echo "copy '$blog_path/public/*' to '/var/www' fail"
    exit 1
fi