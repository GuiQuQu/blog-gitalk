---
title: "搜索"
description: 提高课搜索
date: 2023-05-29T20:34:16+08:00
image:
url: /algorithm-improve/search/
math: false
comments: false
draft: false
categories:
    - algorithm-improve
---

# 提高课搜索

## 双向广搜

双向广搜是广搜的一个加强版本，一般用于求解最小步数的模型中，因为最小步数一般都是指数级别的。
而一般不会用在最短路模型中
通过从头部搜+从尾部搜，然后在中间会和的做法，双向广搜可以大大减少搜索的空间

要点

- 两个方向，都需要有对应的两个队列，并保存对应的搜索步数
- 有两种搜索方式，第一种是简单的一次每一边分别扩展一步，第二种是每次选择元素较少的一方扩展一步，这样比较均衡
- 确定会和点：当我们在一方进行扩展的时候,我可以确定我们需要扩展的下一个点，如果这个点对面方已经扩展到了,那么就可以确定会和了

[例题 is here,字串变换](https://www.acwing.com/problem/content/description/192/)

```cpp
#include<iostream>
#include<algorithm>
#include<unordered_map>
#include<queue>

using namespace std;

const int N = 6;
int n;
string a[N],b[N];

// 扩展时一定还没有相遇
int extend(queue<string> &q,unordered_map<string,int> &da,unordered_map<string,int> &db,string a[],string b[]) 
{
    string str = q.front();
    q.pop();
    // 确定替换开始位置和替换规则
    for (int i = 0; i < str.size(); i++)
        for (int j = 0; j < n; j++) {
            if (str.substr(i,a[j].size()) == a[j]) {
                // substr 只传起始位置会返回到尾部的子串
                // substr 长度超长了会自动截断到尾部
                string state = str.substr(0,i) + b[j] + str.substr(i+a[j].size());
                if (db.count(state)) return da[str] + db[state] + 1;
                if (da.count(state)) continue;
                da[state] = da[str] + 1;
                q.push(state);
            }
        }
    // str经过一步扩展,还是没有和另一端相遇
    return -1;
}

int dfs(string A,string B) {
    if (A == B) return 0;
    queue<string> qa,qb;
    unordered_map<string,int> da,db;
    qa.push(A),qb.push(B);
    da[A] = 0, db[B] = 0;
    // 当一方没有元素(说明这一方已经把能扩展到的点都搜索完了)
    // 并且还没有搜索到答案的时候,说明两者不连通
    while (qa.size() && qb.size()) {
        int t = 0;
        // 每一次选择元素个数较少的一方扩展,这样比较均衡
        if (qa.size() <= qb.size()) t = extend(qa,da,db,a,b);
        else t = extend(qb,db,da,b,a);
        if (t != -1) return t;
    }
    
    return 11;
}

int main() {
    string A, B;
    cin >> A >> B;
    while(cin >> a[n] >> b[n]) n++;
    
    int t = dfs(A,B);
    if (t > 10) printf("NO ANSWER!");
    else printf("%d\n",t);
}
```