---
title: "平衡树"
description: 
date: 2023-09-27T15:48:17+08:00
image:
url: /algorithm-imporve/balanced-tree
math: true
comments: true
draft: false
categories:
---

# 总览

平衡二叉树实际上是动态维护好的一个有序集合,同时尽量保证树的高度是$O(\log n)$

以BST(不保证树的高度)为例,我们可以发现BST的中序遍历结果是有序的

平衡二叉树举例,常见的有红黑树,AVL-树,Treap,splay树等等

cpp中的set和map就是基于红黑树实现的

平衡二叉树的操作一般有
- 插入
- 删除
- 查询key
- 查询前驱，后继
- 查询排名
- 查询比数x小的最大值
- 查询比数x大的最小值

其中map和set支持的操作
- 插入 `insert()`
- 删除 `erase()`
- 查询key `find()` or `count()`
- 查询前驱,后继 `it--` and `it++`
-  `lower_bound(x)` 查询第一个大于等于x的数(>=x min)
-  `upper_bound(x)` 查询第一个大于x的数(>x min)

当我们想要其他操作的时候,往往就需要自己实现平衡树了,使用平衡树推荐写Treap,因为实现起来最简单

当我们在写平衡树的时候,一般会遇到两个旋转操作,分别是左旋和右旋,这两个操作的特点是,
在旋转之后,不会改变原来集合中的顺序,也就是说,中序遍历结果不会变,集合仍然有序。

**左旋**

**右旋**


# Treap
Treap 是BST+堆组成的平衡树。
Treap实际上就是随机二叉树,已经有证明一颗随机二叉树的期望高度为$O(\log n)$。

[例题 is here](https://www.acwing.com/activity/content/problem/content/1726/)

```cpp
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 1e5 + 10, INF = 1e8;
int n;

struct node
{
    int l, r;
    int key, val;
    int cnt;  // 表示这个节点当中有几个值
    int size; // 表示以这个节点为根的子树一共有几个值(包含自己)

} tr[N];

int root, idx;

void pushup(int p) // 汇聚操作,利用子节点信息更新父节点信息(size)
{
    tr[p].size = tr[tr[p].l].size + tr[tr[p].r].size + tr[p].cnt;
}

int get_node(int key)
{
    /**
     * 生成一个新的节点
    {
        l:0
        r:0,
        key:key,
        val:rand(),
        cnt:1,
        size:1
    }
    **/
    tr[++idx].key = key;
    tr[idx].val = rand();
    tr[idx].cnt = 1;
    tr[idx].size = 1;
    return idx;
}

void zig(int &p) // 右旋
{
    int q = tr[p].l;
    tr[p].l = tr[q].r, tr[q].r = p, p = q;
    pushup(tr[p].r), pushup(p);
}

void zag(int &p) // 左旋
{
    int q = tr[p].r;
    tr[p].r = tr[q].l, tr[q].l = p, p = q;
    pushup(tr[p].l), pushup(p);
}

void build()
{
    // 添加两个哨兵节点,这样不用考虑不存在的情况
    get_node(-INF), get_node(INF);
    root = 1, tr[1].r = 2;
    if (tr[1].val < tr[2].val)
        zag(root);
}

// 新节点插入一定会发生在叶节点(BST)
// 为了满足堆的性质,因此需要把val比父节点大的旋转上去
void insert(int &p, int key)
{
    if (!p)
        p = get_node(key);
    else if (tr[p].key == key)
        tr[p].cnt++;
    else if (tr[p].key < key)
    {
        insert(tr[p].r, key);
        if (tr[tr[p].r].val > tr[p].val)
            zag(p);
    }
    else
    {
        insert(tr[p].l, key);
        if (tr[tr[p].l].val > tr[p].val)
            zig(p);
    }
    pushup(p);
}

// 将需要移除的节点旋转到叶子节点,然后直接删除
// 旋转时需要判断把val值高的儿子放到父节点的位置(大根堆)
// 在旋转之后,对应的左子树或者右子树任然是满足(BST+heap的性质的)
void remove(int &p, int key)
{
    if (!p)
        return; // 节点不存在
    if (tr[p].key == key)
    {
        if (tr[p].cnt > 1)
            tr[p].cnt--;
        else if (tr[p].l || tr[p].r) // 非叶节点
        {
            if (!tr[p].r || tr[tr[p].l].val > tr[tr[p].r].val) // 右节点不存在 或者 左节点.val > 右节点.val -> 右旋
            {
                zig(p);
                remove(tr[p].r, key);
            }
            else
            {
                zag(p);
                remove(tr[p].l, key);
            }
        }
        else
            p = 0; // 叶节点
    }
    else if (tr[p].key > key)
        remove(tr[p].l, key);
    else
        remove(tr[p].r, key);

    pushup(p);
}

int get_rank_by_key(int p, int key) // 通过数值找排名
{
    if (!p)
        return 0;
    int l_size = tr[p].l ? tr[tr[p].l].size : 0; // 左子树数值个数

    if (tr[p].key > key)
        return get_rank_by_key(tr[p].l, key);
    else if (tr[p].key == key)
        return l_size + 1;
    else
        return get_rank_by_key(tr[p].r, key) + l_size + tr[p].cnt;
}

int get_key_by_rank(int p, int rank) // 通过排名找数值
{
    if (!p)
        return INF;

    int l_size = tr[tr[p].l].size; // 左子树数值个数

    if (l_size >= rank)
        return get_key_by_rank(tr[p].l, rank);
    else if (l_size + tr[p].cnt >= rank)
        return tr[p].key;
    else
        return get_key_by_rank(tr[p].r, rank - l_size - tr[p].cnt);
}

int get_prev(int p, int key) // 找到严格小于key的最大值
{
    if (!p)
        return -INF;
    if (tr[p].key >= key)
        return get_prev(tr[p].l, key);
    else
        return max(tr[p].key, get_prev(tr[p].r, key));
}

int get_next(int p, int key) // 找到严格大于key的最小值
{
    if (!p)
        return INF;
    if (tr[p].key <= key)
        return get_next(tr[p].r, key);
    else
        return min(tr[p].key, get_next(tr[p].l, key));
}

int main()
{
    build();

    scanf("%d", &n);
    while (n--)
    {
        int opt, x;
        scanf("%d%d", &opt, &x);
        if (opt == 1)
            insert(root, x);
        else if (opt == 2)
            remove(root, x);
        else if (opt == 3)
            printf("%d\n", get_rank_by_key(root, x) - 1);
        else if (opt == 4)
            printf("%d\n", get_key_by_rank(root, x + 1));
        else if (opt == 5)
            printf("%d\n", get_prev(root, x));
        else
            printf("%d\n", get_next(root, x));
    }
    return 0;
}
```
