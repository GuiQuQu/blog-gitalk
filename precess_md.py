# 考虑过使用其他渲染引擎,但是其他渲染引擎又有自己的问题,最终决定自己手动来解决这个问题
# 预处理content文件夹下的所有的md文档,给所有公式块添加,来让hugo的默认md引擎正确渲染公式,同时保证自己的阅读和书写体验
# {{ math }}
# {{ /math }}
import os
import re
from pathlib import Path
math_indu = {
    "first": "{{< math >}}",
    "second": "{{< /math >}}"
}

target_dir = "./content/post"

def process_md(md_path:str,save_path):
    new_md = ""
    f = open(md_path,mode="r",encoding="utf-8")
    line = f.readline()
    while line:
        
        if line.strip() == math_indu["first"]:
            # Case 1 已经完成了处理
            while line and line.strip() != math_indu["second"]:
                new_md += line
                line = f.readline()
            new_md += line
        elif line.strip() == "$$":
            # Case 2 处理多行公式
            new_md = new_md + math_indu["first"] + "\n"
            # add $$
            new_md += line
            line = f.readline()
            while line and line.strip() != "$$":
                new_md += line
                line = f.readline()
            new_md += line
            new_md = new_md + math_indu["second"] + "\n"
        else:
            # Case 3 处理行内公式
            re_str = r"(?<!{{< math >}})(\$.*?\$)(?!{{< \/math>}})"
            result = re.finditer(re_str, line)
            for item in result:
                match = item.group(0)
                line = line.replace(match, math_indu["first"] + match + math_indu["second"])
            new_md += line
        line = f.readline()
    f.close()
    cout = open(save_path,mode="w",encoding="utf-8")
    cout.write(new_md)
    cout.close()
    
def process(path:str):
    for p in os.listdir(path):
        curp = os.path.join(path,p)
        if os.path.isdir(curp):
            process(curp)
        else:
            path_str = Path(curp)
            if path_str.suffix != ".md":
                continue
            if path_str.stem == "_index" or path_str.stem == "index":
                continue
            save_path = os.path.join(path_str.parent,"index.md")
            process_md(curp, save_path)
            break

def main():
    process(target_dir)

if __name__ == "__main__":
    main()


