---
title: "贪心算法"
description: 
date: 2023-05-29T20:20:22+08:00
image:
url: /algorithm-base/greedy/
math: true
comments: false
draft: false
categories:
    - "algorithm-base"
---

# 贪心

## 区间问题

### 区间选点

[题目链接 is here](https://www.acwing.com/problem/content/description/907/)

贪心求解做法

```
按照右端点从小到大排序
从前到后遍历
	if 当前区间已经被包含 pass
	else 选择该区间的最右端点

证明:
	设最优解是ans,采用以上解法得到解是cnt
	(1)证明ans<=cnt
	因为cnt是一个可行解,所以ans<=cnt
	(2)证明ans>=cnt
	按照以上方法会去确定cnt个区间，(如果包含就求交集)，不包含cnt++，
	这些区间互不相交，至少需要cnt个点采用全部选到，所以必须要求ans>=cnt
```

代码

```cpp
#include<iostream>
#include<algorithm>

using namespace std;

const int N = 100010;

struct Range
{
    int l,r;
    bool operator< (const Range &W)const
    {
        return r < W.r;
    }
} f[N];

int n;

int main()
{
    cin>>n;
    for (int i=0;i<n;i++)
    {
        int a,b;
        cin>>a>>b;
        f[i] ={a,b};
    }
    sort(f,f+n);
    int res =0;
    int ed =-2e9;
    for (int i=0;i<n;i++)
    {
        if (f[i].l>ed)
        {
            res++;
            ed = f[i].r;
        }
    }
    cout<<res<<endl;
}
```

### 最大不相交区间的数量

题目链接:https://www.acwing.com/problem/content/910/

贪心做法

```
按照右端点从小到大排序
从前到后遍历
	if 当前区间已经包含点 则直接pass
	else 选择当前区间的右端点
	
证明: ans,最优解  cnt,贪心解
(1) ans>=cnt,因为按照以上选法，选出来的区间,一定是不重复的，所以ans>=cnt
(2) ans<=cnt,按照以上方法,可以选出cnt个点,这cnt个点可以覆盖所有的区间,
	每一个点最多对应一个区间，所以最多只能选出cnt个区间。否则一定会出来相交的区间，所以ans<=cnt
```

代码

```
和上一题代码一样
```

### 区间分组

​	又名区间图着色问题，我们可以构造一个区间图，顶点表示所有区间(活动)，用边连接所有不兼容的活动，然后问你最少需要多少种颜色，可以将所有顶点全部染色，并且相邻节点颜色都不相同。

题目链接:https://www.acwing.com/problem/content/908/

贪心做法

```
按照左端点从小到大排序
从前向后遍历
	if 该区间可以放到前面的组里，则放到前面的组里
	else 开一个新组
如何判定是否可以放到前面的组里:记录每一个组最右边端点,如果该区间的左端点小于所有组区间的右端点,则不能放到任何一个组里去，必须开新组(因此只需要保存最小的组的右端点即可)

证明,ans是最优解，是最少的组，cnt是按照上面这种做法得到的结果
1.ans<=cnt，因为ans是最优解，cnt是可行解，所以ans<=cnt
2.ans>=cnt,即证明最少需要选择cnt个组，按照以上方法进行选择时，一定需要cnt个组，考虑当前有cnt-1个组,新的区间因为小于所有组区间的右端点，会和每一个组冲突，必须开一个新组采用符合题意。因此至少需要cnt个组才能将上面的区间分开。
```

代码

```cpp
#include<iostream>
#include<queue>
#include<algorithm>
#include<vector>

using namespace std;

const int N = 100010;

struct Range
{
    int l,r;
    bool operator< (const Range &W)const
    {
        return l<W.l;
    }
} range[N];

int main()
{
    int n;
    scanf("%d",&n);
    for (int i=0;i<n;i++)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        range[i] = {a,b};
    }
    sort(range,range+n);
    priority_queue<int,vector<int>,greater<int>> q;
    
    for (int i =0;i<n;i++)
    {
        if (q.size() == 0 || range[i].l<=q.top()) q.push(range[i].r); //range[i]和前面的组有重叠
        else
        { //range[i]和前面的右端点最小的组没有重叠，可以直接向后加
            int t= q.top();
            q.pop();
            q.push(range[i].r);
        }
    }
    cout<< q.size() <<endl;
}
```

### 区间覆盖

题目链接:https://www.acwing.com/problem/content/909/

贪心做法

```
按照左端点从小到大排序
从前向后枚举每一个区间
	在所有可以覆盖左端点的区间中，选择右端点最大的区间，然后将st更新为该区间的右端点

证明:
	找到最优解和贪心解第一个不同的区间,因为贪心解是能覆盖当前st的右端点最大的区间,st是上一选中区间的右端点或者要求覆盖区间的左端点，所以可以用贪心解代替最优解的区间，仍是最优解(不亏)。在替换完之后，最优解的下一个区间，可能被完全覆盖，也可能左端点被包括，右端点伸出来，如果被完全覆盖，可以直接删除，如果不被完全覆盖，使用贪心解来替换最优解。重复下去。所以可以得到cnt<=ans,又因为ans是最优解，所以ans>=cnt。所以ans==cnt
```

代码

```cpp
#include<iostream>
#include<algorithm>

using namespace std;

const int N = 100010;

struct Range
{
    int l,r;
    bool operator< (const Range &W)const
    {
        return l<W.l;
    }
}range[N];

int main()
{
    int st,ed;
    scanf("%d%d",&st,&ed);
    int n;
    scanf("%d",&n);
    for (int i=0;i<n;i++)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        range[i] ={a,b};
    }
    sort(range,range+n);
    
    int res=0;
    bool success =false;
    for (int i=0;i<n;i++)
    {
        int j=i,r =-2e9;
        //双指针来找所有能覆盖st的右端点最长的区间
        while(j<n && range[j].l<=st)
        {
            r =max(r,range[j].r);
            j++;
        }
        if (r<st) //没有任何区间可以覆盖st
        {
            res = -1;
            break;
        }
        res++;
        st =r;  //更新st
        if (r>=ed) //已经找到答案,还有可能最后ed无法覆盖
        {
            success=true;
            break;
        }
        i=j-1;
    }
     if (!success) res =-1;
     cout<<res<<endl;
}
```

## 哈夫曼树

### 合并果子

[题目链接](https://www.acwing.com/problem/content/150/)

哈夫曼贪心，合并果子就是模板题，做法是每次选择合并代价最小的两个节点进行合并，删除这两个节点，加入新的节点，合并代价是两者之和，然后重复这个过程，知道只剩一个节点。可以用堆实现。

模板题代码

```cpp
#include<iostream>
#include<queue>
#include<algorithm>
#include<vector>

using namespace std;

int n;
int main() 
{
    scanf("%d",&n);
    priority_queue<int, vector<int>, greater<int>> q;
    for (int i = 0; i < n; i ++)
    {
        int a;
        scanf("%d",&a);
        q.push(a);
    }
    int res = 0;
    while (q.size() >= 2) 
    {
        int t1 = q.top();
        q.pop();   
        int t2 = q.top();
        q.pop();
        res += (t1 + t2);
        q.push(t1 + t2);
    }
    printf("%d", res);
    return 0;
}
```

证明先略过

## 排序不等式

设有两个序列$a_1,a_2,...,a_n$和$b_1,b_2,...,b_n$，这两者数量均从小到大排好序，数列$c_1,c_2,...,c_n$是$b_1,b_2,...,b_n$的乱序排列，那么有以下不等式成立
$$
a_1*b_n+a_2*b_{n-1}+...+a_n *b_1 <=a_1*c_1+a_2*c_2+...+a_n*c_n<=a_1*b_1+a_2*b_2+...a_n*b_n\\
即\sum_{i=1}^{n}a_i*b_{n-i+1}<=\sum_{i=1}^{n}a_i*c_i<= \sum_{i=1}^{n}a_i*b_i
$$

### 排队打水

[题目链接](https://www.acwing.com/problem/content/description/915/)

可以选择使用排序不等式直接解题，

形成序列后，考虑每个人打水让后面的等待的时间，因此第一个的人打水的时间后面所有人都得等待，因此需要放最小的打水时间上去。

这里会具体问题具体分析

贪心解是优先让打水时间最短的人打水

证明

```
设优化解是的打水序列是
a1 a2 ....
贪心解的打水序列是
b1 b2 ....
元素ai的等待时间是[a1+a2+...+a_(i-1)]
从第一个优化解和贪心解不同的位置开始，设该位置为i
a1 a2 a3 ... ai a_(i+1) ... bi ...
b1 b2 b3 ... bi b_(i+1) ...

交换ai和bi,这时计算优化解的总打水时间:bi后的不变,ai前的不变
有bi <= ai，那么在ai和bi之间的元素的每个人单独的等待时间中把ai换成了bi,不增加了ai到bi之间的人的等待时间
```

代码

```cpp
#include<iostream>
#include<algorithm>

using namespace std;

const int N = 1e5 + 10;

int f[N];
int n;
int main () 
{
    scanf("%d", &n);
    
    for (int i = 0; i < n; i++)
        scanf("%d", &f[i]);
        
    sort(f, f + n);
    
    long long res = 0;
    int t = 0;
    for (int i = 0; i < n; i ++) 
    {
        res += t;
        t += f[i];
    }
    cout << res << endl;
    return 0;
}
```

## 绝对值不等式

### [货仓选址](https://www.acwing.com/problem/content/106/)

贪心思路是将位置从大到小排序后，选择最中间的位置，奇数则定位中心点，偶数则可以在两者最中心的点之间选择任意一个位置(闭区间内)。

计算最短距离和是，$(a_1,a_n),(a_2,a_{n-1})$做一对，然后依次进行类推，这是按照上面的方法进行选择，货仓到一对点的距离和就是这对点之间的距离，而如果要把货仓选择别处，仍旧可以把该点框住的点对距离不变，但是框不住的点对，货仓到这个点对的距离之和就会大于该店对之间的距离，导致解变大。

```cpp
#include<iostream>
#include<algorithm>

using namespace std;

const int N = 100010;

int f[N];
int n;

int main() 
{
    scanf("%d", &n);
    for (int i = 0; i < n; i++)
        scanf("%d", &f[i]);
    
    sort(f, f + n);
    int i = 0, j = n - 1;
    long long res = 0;
    while (i < j) 
    {
        res += (f[j] - f[i]);
        j--, i++;
    }
    printf("%ld",res);
    
    return 0;
}
```

## 推公式

### [耍杂技的牛](https://www.acwing.com/problem/content/127/)

贪心做法是按照`wi+si`从小到大排序，越小的放在越上面。

证明采用邻项交换法，设有两个头牛挨着,$(w_i,s_i)$在$(w_{(i+1)},s_{(i+1)})$的上面。

![image-20220804222448402](yxc讲解笔记.assets/image-20220804222448402.png)

从上面的推导得到的结论是，当$w_i + s_i > w_{(i+1)} + s_{(i+1)}$的时候，把i+1放到前面去，不会导致该方案的最大风险值变大。这对任意一个方案都是有效的，因此也包括最优方案，所以对于最优方案，我们仍然可以如此交换，从而得到一个按照s+w升序排序的序列，最优解也在里面。

```cpp
#include<iostream>
#include<algorithm>

using namespace std;

const int N = 50010;

typedef pair<int,int> PII;

PII cows[N];
int n;

bool cmp(const PII &a, const PII &b)
{
    return a.first + a.second < b.first + b.second;
} 

int main () 
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++) 
    {
        int w, s;
        scanf("%d%d",&w,&s);
        cows[i] = {w, s};
    }
    sort(cows,cows + n, cmp);
    int res = -2e9, sum = 0;
    for (int i = 0; i < n; i++)
    {
        res = max(res, sum - cows[i].second);
        sum += cows[i].first;
    }
    cout << res << endl;
    return 0;
}
```