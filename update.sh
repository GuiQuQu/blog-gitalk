# !/bin/bash
# the shell script is used to update blog content everyday
# please add the script to crontab,and user is root

blog_path="/root/guiququ-myblog"
cd $blog_path
# get blog content from git
/usr/bin/git pull
if [ $? != "0" ]; then
    echo "git pull fail"
    exit 1
fi
# handle md files
echo "start handle md files (used python3)"
/usr/bin/python3 precess_md.py
if [ $? != "0" ]; then
    echo "precess_md.py fail"
    exit 1
fi
# build blog
$GOPATH/bin/hugo
if [ $? != "0" ]; then
    echo "hugo build fail"
    exit 1
fi
# copy build content to '/var/www'
cp -r $blog_path/public/* /var/www
if [ $? != "0" ]; then
    echo "copy '$blog_path/public/*' to '/var/www' fail"
    exit 1
fi