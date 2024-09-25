---
title: "Algorithm杂项内容"
description: 
date: 2024-03-31T15:56:41+08:00
image:
url:
math: true
comments: true
draft: false
categories:
---

# 点集的最大曼哈顿距离



因为$|a-b| = max(a-b,b-a)$

所以曼哈顿距离 $d = |x_1-x_2| + |y_1-y_2|$ 可以拆成 $d = max(x_1-x_2,x_2 -x_1)+max(y_1-y_2,y_2 - y_1)$

因为需要从两个max括号中选择,因此我们可以直接枚举出来

$$
\begin{aligned}
        d & = max(x_1-x_2,x_2 -x_1)+max(y_1-y_2,y_2 - y_1) \\ 
        &  = max(\\
        & \quad x_1-x_2 + y_1-y_2, \\
        & \quad x_1-x_2 + y_2-y_1, \\
        & \quad x_2-x_1 + y_1-y_2, \\
        & \quad x_2-x_1 + y_2-y_1 \\
        & )
\end{aligned}
$$

进行整理,可以得到

$$
\begin{aligned}
        d & = max(x_1-x_2,x_2 -x_1)+max(y_1-y_2,y_2 - y_1) \\ 
        &  = max(\\
        & \quad (x_1+y_1) - (x_2+y_2), \\
        & \quad (x_1-y_1) - (x_2-y_2), \\
        & \quad (x_2 - y_2) -(x_1-y_1), \\
        & \quad  (x_2 + y_2) -(x_1 + y_1) +\\
        & ) \\
        & = max( \\
        & |x_1 + y_1| - |x_2 + y_2|, \\
        & |x_1 - y_1| - |x_2 - y_2|, \\
        & )
\end{aligned}
$$

最后推导出来的就是曼哈顿距离和切比雪夫距离之间的关系,
两者可以相互转换。

切比雪夫距离的定义:

切比雪夫距离（Chebyshev distance）是向量空间中的一种度量，二个点之间的距离定义为其各坐标数值差的最大值。

在二维空间内，两个点之间的切比雪夫距离为它们横坐标之差的绝对值与纵坐标之差的绝对值的最大值。设点 $A(x_1,y_1),B(x_2,y_2)$，则 $A,B$ 之间的切比雪夫距离用公式可以表示为：
$$
    d(A,B) = max(|x_1-x_2|,|y_1-y_2|)
$$


例题: [leetcode-100240. 最小化曼哈顿距离](https://leetcode.cn/problems/minimize-manhattan-distances/)

这道题需要先求解一个点集的最大曼哈顿距离,我们可以将曼哈顿距离转换为切比雪夫距离。当转换完之后,最大的曼哈顿距离从转化之后点的最大横坐标差值和最大纵坐标差值之间产生。
因此我们可以分别排序`x+y`和`x-y`,然后用最大减最小,就可以求出最大的曼哈顿距离。

然后题目要求我们移除一个点之后，曼哈顿距离的最小值，要想最大曼哈顿距离变小，我们必须移除原来组成最大曼哈顿距离的点,我们可以枚举这些点,然后在重新计算一次最大曼哈顿距离。

示例代码
```cpp
typedef pair<int,int> PII;
class Solution {
public:
    int ans = 1e9+10;
    int minimumDistance(vector<vector<int>>& points) {
        vector<int> rms = get_max(points,-1);
        for (int i : rms) 
            get_max(points,i);
        return ans;
    }

    // 返回和最大曼哈顿距离有关的4个点的下标
    vector<int> get_max(vector<vector<int>> & points, int rm) {
        vector<PII> a,b;
        for (int i = 0; i < points.size(); i++) {
            if (i != rm) {
                auto &p = points[i];
                a.push_back({p[0]+p[1],i});
                b.push_back({p[0]-p[1],i});
            }
        }
        sort(a.begin(), a.end());
        sort(b.begin(),b.end());
        int d = max(a.back().first - a[0].first, b.back().first - b[0].first);
        ans = min(ans,d);
        return vector<int> {a.back().second, a[0].second, b.back().second, b[0].second};
    }
};
```