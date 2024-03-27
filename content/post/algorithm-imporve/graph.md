---
title: "图算法"
description: 
date: 2024-03-24T22:52:06+08:00
image:
url:
math: true
comments: true
draft: false
categories:
---

# LCA(最近公共祖先)

LCA是给定树上的任意两个点,求解他们的最近公共祖先

方法1:向上标记法

首先找到所有节点的父节点,然后记录一个节点向上走走到父节点的路径,然后让另一个节点向上走,如果走到了那条路径上,那么那个交点就是最近公共祖先

时间复杂度 $O(n)$

方法2:倍增法

倍增法是一种在线做法

1. 首先预处理出来所有点向上跳2^j步的点是谁`fa[i][j]` 从点i跳2^j步,以及所有点的深度数组depth

2. 对于两个点u,v,设定`depth[u] < depth[v]`,先让u跳到和v一样的深度,然后同时往上跳,直接跳到他们的最近公共祖先的子节点,然后通过fa数组找到最近公共祖先

3. 如何预处理fa数组, 首先 `fa[i][0]`是往上跳一步,因此是父节点,然后我们可以通过递推的方式求解其他解(先跳一半 `t = fa[i][j-1]`,然后在跳一半`fa[t][j-1]`),`fa[i][j] = fa[fa[i][j-1]][j-1]`

4. 如何跳? 采用二进制优化的思想,假设`depth[v] - depth[u] = k`, 我们可以将k按二进制位分解,为1则跳对应的步数,为零则不跳,在具体实现的时候不会把k求解出来
   
可以这样写: 从v开始跳,从大到小遍历二次幂的可以跳的步数,如果`depth[fa[v][j]] >= depth[u]`,说明`k`的对应位就是1,我们可以往上跳

4. 如何同时往上跳,同样是从大到小遍历二次幂可以跳的步数,如果`fa[u][j] != fb[b][k]`,说明还没有跳到他们的最近公共祖先,我们可以继续往上跳,不使用`fa[u][j] == fb[b][k]`的原因是当这个条件成立时,我们仅仅知道该点是公共祖先,但是不确定是否是最小公共祖先,按照这种方式我们会一直走到最小公共祖先的孩子节点。

跳t步,跳到孩子节点(跳过了(跳的步数>t)的条件就是`fa[u][j] == fb[b][j]`),然后将t做二进制分解

5. 设置哨兵节点,我们可以设置0号点为哨兵节点,`depth[0] = 0`,当节点跳了出根节点之后,就返回哨兵节点

6. 预处理用dfs或者bfs都可以,这里的示例使用bfs

时间复杂度分析

1. 预处理需要遍历所有节点+每个节点二进制跳步计算,时间复杂度$O(n \log_2 n)$,每次查询的时间复杂度$O(\log_2 n)$

在多次查询的情况下,倍增法的时间复杂度要优于向上标记法

示例代码

```cpp
#include<iostream>
#include<queue>
#include<cstring>

using namespace std;

const int N = 4e4 + 10, M = 2*N;

int h[N],e[M],ne[M],idx;

int depth[N];
int fa[N][18];

int n,m;
void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx++;
}

void bfs(int root) {
    queue<int> q;
    depth[0] = 0, depth[root]=1;
    fa[root][0] = 0;
    q.push(root);
    while (q.size()) {
        int u = q.front(); q.pop();
        
        for (int i = h[u]; i != -1; i = ne[i]) {
            int j = e[i];
            if (!depth[j]) {
                depth[j] = depth[u] + 1;
                q.push(j);
                // 预处理 fa[j][k]
                fa[j][0] = u;
                for (int k=1; k <= 16; k++)
                    fa[j][k] = fa[fa[j][k-1]][k-1]; // 处理跳过头的情况依靠fa[root][0] = 0;
            }
        }
    }
}

int lca(int a, int b) {
    
    if (depth[a] < depth[b]) swap(a,b);
    for (int k = 15; k >= 0; k--)
        // 通过哨兵节点depth是0,可以避免跳过根
        if (depth[fa[a][k]] >= depth[b])
            a = fa[a][k];
            
    if (a == b) return a;
    
    for (int k = 15; k >= 0; k--)
        // 通过哨兵节点均是0,可以避免跳过根
        if (fa[a][k] != fa[b][k]) 
        {
            a = fa[a][k];
            b = fa[b][k];
        }
    return fa[a][0];
}

int main() {
    cin >> n;
    int root;
    memset(h,-1,sizeof h);
    for (int i = 0; i < n; i++) {
        int a,b;
        cin >> a >> b;
        if (b == -1) root = a;
        else add(a,b), add(b,a);
    }
    bfs(root);
    cin >> m;
    while(m--) {
        int x, y;
        cin >> x >> y;
        int p = lca(x,y);
        if (p == x) puts("1");
        else if (p == y) puts("2");
        else puts("0");
    }
}
```
   