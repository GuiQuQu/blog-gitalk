---
title: "C++ 结构体,类以及STL容器的基本用法"
description: 
date: 2023-09-25T15:29:15+08:00
image:
url:
math: false
comments: false
draft: false
categories:
    - algorithm-base
---

# 结构体

**定义一个结构体**
```cpp
struct Student{
    int id;
    string name;
    int age;
};
```
**声明一个结构体**

```cpp
Student stu;
stu.id = 1;
stu.name = "Tom";
stu.age = 18;
```
**声明一个结构体,并完成初始化**
```cpp
Student stu = {1,"Tom",18};
```
**使用new关键字声明结构体**

注意使用new关键字得到的是指针,所以要使用->来访问结构体的成员
```cpp 
Student *stu = new Student;
stu->id = 1;
stu->name = "Tom";
stu->age = 18;
```
**添加构造函数,像声明类一样声明结构体**
```cpp
struct Student{
    int id;
    string name;
    int age;
    Student(int _id, string _name, int _age): 
    id(_id),name(_name),age(_age) {}
};

int main() {
    Student stu1 = Student(1,"Tom",18);
    Student* stu2 = new Student(2,"Jack",19);
}
```

**在结构体内重载运算符**
重载运算符可以修改元素是结构体的stl容器的排序方式
```cpp
struct Student{
    int id;
    string name;
    int age;

    bool operator< (const Student &a) const{
        return id < a.id;
    }
};
```

# 函数

**参数传递:传值**
```cpp
void func(int a) {
    printf("a = %d\n",a);
}

int main() {
    func(10);
}
// Output
// a = 10
```

**参数传递:传引用**
```cpp
void func(int &a) {
    a = 20;
}

int main() {
    int a = 10;
    func(a);
    printf("a = %d\n",a); 
}
// Output
// a = 20
```

**参数传递:传指针**
```cpp
void func(int *a) {
    *a = 20;
}

int main() {
    int a = 10;
    func(&a);
    printf("a = %d\n",a);
}
// Output
// a = 20
```
**函数返回:返回非复合类型**
```cpp
string func() {
    return "hello world";
}

int main() {
    string str = func();
    cout << str << endl;
}
```

**重点：不要返回局部对象的引用或者指针**
因为在函数结束之后,局部对象申请的内存也会被释放,因此函数种植意味着局部变量的引用或者指针将指向不在有效的内存空间


**函数返回:返回引用类型**
```cpp
struct Student {
    int id;
    string name;
    int age;
};

Student &func1(Student &stu, int id) {
    stu.id = id;
    return stu;
}

void func2(Student &stu, int id) {
    stu.id = id;
}

int main() {
    Student stu1 = {1,"Tom",18};
    Student stu2 = func1(stu1,20);
    printf("stu1.id=%d\tstu2.id=%d\n",stu1.id,stu2.id);
    func2(stu1,30);
    printf("stu1.id=%d\tstu2.id=%d\n",stu1.id,stu2.id);
}
// Output
//stu1.id=20      stu2.id=20
//stu1.id=30      stu2.id=20
```

# 类

**类和结构体的区别**
结构体是一种特殊的类,在结构体内直接声明的变量默认的访问权限是public,而在类内声明的变量默认的访问权限是private

**定义类**
```cpp
struct Sales_data {

    string bookNo;
    unsigned units_sold = 0;
    double revenue = 0.0;
    
    // 内联函数(inline function)
    string isbn() const {return bookNo;}
    
    // 返回this对象的函数
    Sales_data& combine(const Sales_data&);

    // 常量成员函数
    // 带const声明的函数,在该函数内,不能修改类的成员变量
    double avg_price() const;
};
```
在类的成员函数中,可以通过关键字`this`得到该对象示例的地址,因此可以用类似于`this->bookNo`的方式在类内访问成员变量

`this`是一个常量指针,总是指向当前对象实例,我们不能改变`this`的指向

当在成员函数后加入`const`之后,`this`的指针类型会被改变成指向常量的指针,因此在常量成员函数中,不能修改类的成员变量

*在类的外部定义成员函数*
```cpp
double Sales_data::avg_price() const {
    if (units_sold) {
        return revenue/units_sold;
    } else {
        return 0;
    }
}
```

*返回this对象的函数的实现*
```cpp
Sales_data& Sales_data::combine(const Sales_data &rhs) {
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}
```

*类的构造函数*
```cpp

struct Sales_data {
    string bookNo;
    unsigned units_sold = 0;
    double revenue = 0.0;

    // 默认的构造函数
    Sales_data() = default;
    // 构造函数初始值列表
    Sales_data(const string &s): bookNo(s) {}
    Sales_data(const string &s, unsigned n, double p):
        bookNo(s), units_sold(n), revenue(p*n) {}
    // 在类的外部定义的构造函数
    Sales_data(istream &);
};
```

# STL容器

**容器通用操作总结**

成员类型
- `iterator` 此容器的迭代器类型
- `const_iterator` 此容器的常量迭代器类型

使用例:`vector<int>::iterator it = v1.begin();`
- `size_type` 无符号整数类型,用于表示容器大小
- `difference_type` 带符号整数类型,足够保存两个迭代器之间的距离
- `value_type` 元素类型

构造函数
- `C c;` 默认构造函数,构造空容器,如果c是一个`array`,则c中元素按默认方式初始化,否则c为空
- `C c1(c2);` or `C c1 = c2` 构造c2的拷贝c1,要求c1和c2具有相同的类型(即c1和c2的元素类型和容器类型都相同,对于`array`来说,还要求大小相同)
- - `C c{a,b,c...}` or `C c={a,b,c,...}` 用列表初始化c,要求列表中元素类型必须和C的元素类型相容,对于`array`类型,列表中元素数据必须等于或者小于`array`的大小,任何遗漏的元素都进行值初始化
- `C c(b,e)` 构造c, 将迭代器b和e指定的范围内的元素拷贝到c(array不支持)
- `C seq(n)` *只有顺序容器支持*, seq包含n个元素,这些元素进行值初始化
- `C seq(n,t)` *只有顺序容器支持*, seq包含n个元素,每个元素的值都是t

赋值与swap
- `c1 = c2` 将c1中的元素替换为c2中的元素
- `c1 = {a,b,c...}` 用列表中的元素替换c1中的元素
  
  `swap`操作不移动元素,仅交换两者的数据结构,除`string`外,因此指向容器的迭代器,引用和指针在swap操作后都不会失效
- `c1.swap(c2)` 交换c1和c2中的元素
- `swap(c1,c2)` 交换c1和c2中的元素

大小
- `c.size()` 返回c中元素的个数(不支持forward_list)
- `c.empty()` 如果c中没有元素,返回true,否则返回false
- `c.max_size()` 返回c能容纳的最大元素数量
  
添加/删除元素
注:不同容器中,这些操作的接口都不同
- `c.insert(args)` 将args中的元素拷贝进c
- `c.emplace(args)` 使用args构造一个元素,插入c中
- `c.erase(args)` 删除args指定的元素
- `c.clear()` 删除c中的所有元素,return void

关系运算符
- `==`,`!=` 所有容器都支持
- `<`,`<=`,`>`,`>=` 关系运算符(无序容器不支持)
  
获得迭代器
- `c.begin()`, `c,end()` 返回指向容器c中第一个元素和尾元素之后位置的迭代器
- `c.cbegin()`, `c.cend()` 返回const_iterator

反向容器的额外成员(不支持forward_list)
- `reverse_iterator` 按逆序寻址元素的迭代器
- `const_reverse_iterator` 按逆序寻址元素的常量迭代器
- `c.rbegin()`, `c.rend()` 返回指向容器c中最后一个元素和首元素之前位置的迭代器
- `c.crbegin()`, `c.crend()` 返回const_reverse_iterator

## 顺序容器
- vector 变长数组 `#include<vector>`
- deque 双端队列 `#include<deque>`
- list 双向链表 `#include<list>`
- forward_list 单向链表 `#include<forward_list>`
- array 固定大小数组 `#include<array>`
- string 与vector相似的容器,但专门用于保存字符
  
### 向顺序容器中添加元素


(array不支持添加元素)

`forward_list`不支持`push_back`和`emplace_back`
`forward_list`有自己专有版本的`insert`和`emplace`
`vector`和`string` 不支持`push_front`和`emplace_front`
- `c.push_back(t)` or `c.emplace_back(args)` 在c的尾部创建一个值为t的元素或者是由args创建的元素,返回void
- `c.push_front(t)` or `c.emplace_front(args)` 在c的头部创建一个值为t的元素或者是由args创建的元素,返回void
- `c.insert(p,t)` or `c.emplace(p,args)` 在迭代器p指向的元素之前常见一个值为t或者由args创建的元素,返回指向新添加元素的迭代器
- `c.insert(p,n,t)` 在迭代器p指向的元素之前插入n个值为t的元素,返回指向第一个新添加元素的迭代器,若n为0,则返回p
- `c.insert(p,b,e)` 将迭代器b和e指向的范围内的元素插入到迭代器p指向的元素之前。b和e不能指向c中的元素,返回指向第一个新添加元素的迭代器,若范围为空,则返回p
- `c.insert(p,il)` `il`是一个花括号包围的元素值范围,返回值行为同上
例如
```cpp
vector<string> v1{"hello","world"};
auto it1 = v1.insert(v1.end(),{"a","b","c"});
for (auto it =it1; it != v1.end(); it++) {
    cout << *it << endl;
}
//output
// a b c
```
**push操作和emplace操作的区别**
使用push操作会拷贝元素,而使用emplace操作,则会将参数传递给元素类型的构造函数,emplace成员使用这些参数在容器管理的内存空间中直接构造函数

### 访问元素
at和下标操作只适用于string和vector,deque和array

back不适用于forward_list

- `c.back()` 返回c中尾元素的一个引用
- `c.front()` 返回c中首元素的一个引用
- `c[n]` 返回c中第n个元素的一个引用,不执行边界检查
- `c.at(n)` 返回c中第n个元素的一个引用,执行边界检查,如果n越界,抛出一个`out_of_range`异常

**注意,访问成员函数返回的是引用**

### 删除元素
`array`不适用

`forward_list`不支持`pop_back`

`vector`和`string`不支持`pop_front`

- `c.pop_back()` 删除c中的尾元素,返回void
- `c.pop_front()` 删除c中的首元素,返回void
- `c.erase(p)` 删除迭代器p指向的元素,返回指向被删元素之后元素的迭代器
- `c.erase(b,e)` 删除迭代器b和e指定的范围内的元素,返回e
- `c.clear()` 删除c中的所有元素,返回void

删除元素不检查元素是否存在,需要程序员自己检查

### 容器操作可能使迭代器失效
**待补充**
使用迭代器修改元素值是有效的

在向容器添加元素之后
- `vector`和`string`的迭代器可能失效
- `deque`的迭代器可能失效,但不会失效
- `list`和`forward_list`的迭代器不会失效

当我们从一个容器中删除元素后,指向被删除元素的迭代器,指引和引用会失效

## 容器适配器
- `stack`
- `queue`
- `priority_queue`

### 栈
`#include<stack>`

栈默认基于deque实现,也可以在list或者vector上实现
```cpp
stack<int> stk1;
stack<int,vector<int>> stk2;
```
- `s.pop()` 删除栈顶元素,返回void
- `s.push(item)`  or ``s.emplace(args)` 在栈顶创建一个值为item或者由args创建的元素,返回void
- `s.top()` 返回栈顶元素

### 队列和优先队列
`#include<queue>`
queue和priority_queue定义在`queue`头文件中

注意：`queue`不能使用vector做底层容器

queue默认基于deque实现,priority_queue默认基于vector实现
- `q.pop()` 出队
- `q.front()` or `q.back()` 队首or队尾,只适用于queue
- `q.top()` 返回堆顶,只适用于priority_queue
- `q.push(item)` or `q.emplace(args)` 入队

priority_queue默认是大根堆,如果需要定义小根堆,采用以下方式,或者使用用大根堆存负值
`priority_queue<int,vector<int>,greater<int>> pq;`

## 泛型算法
`#include<algorithm>`

- `sort(b,e,cmp)` 对迭代器b和e指定的范围内的元素进行排序,cmp是一个可调用对象,用于比较元素,默认使用<运算符
```cpp
bool isShorter(const string &s1, const string &s2) {
    return s1.size() < s2.size();
}
int main() {
    vector<string>v1{"a","abc","123","cd"};
    sort(v1.begin(),v1.end(),isShorter);
}
```
- `stable_sort(b,e,cmp)` 稳定排序
- `unique(b,e)` 消除重复元素,返回指向不重复区域之后一个位置的迭代器

## 关联容器

关联容器(有序和无序)都支持前面介绍的普通容器操作

按关键字有序保存元素
- map `#include<map>`
- set `#include<set>`
- multimap `#include<map>`
- multiset `#include<set>`

无序集合
- unordered_map `#include<unordered_map>`
- unordered_set `#include<unordered_set>`
- unordered_multimap `#include<unordered_map>`
- unordered_multiset `#include<unordered_set>`

有序容器要求关键字严格弱序

无序容器要求关键字可哈希

### pair类型

- `pair<T1,T2> p(v1,v2)` or `pair<T1,T2> p = {v1,v2}`用v1和v2初始化p
- `make_pair(v1,v2)` 用v1和v2初始化一个`pair`对象,pair的类型从v1和v2推导而来
- `p.first` 返回p的第一个元素
- `p.second` 返回p的第二个元素
- `p1 relop p2` 关系运算符,对pair进行字典序比较(先比较first在比较second)
- `p1 == p2` or `p1 != p2` 当first和seccond都相等时,两个pair相等,否则不等

### 有序容器

#### 关联容器额外的类型别名

- `key_type` 关键字类型
- `mapped_type` 映射类型(只适用于map)
- `value_type` `set`中的关键字类型等于`key_type`,而`map`中的关键字类型等于`pair<const key_type,mapped_type>`

#### 关联容器迭代器

当解引用一个关联容器迭代器时,我们会得到一个类型为容器的`value_type`的值的引用,对map来说是一个pair，first保存key,second保存value。

利用map的迭代器,我们可以改变value,但是不能改变key

对set来说是一个关键字

利用set的迭代器,我们不能改变关键字

#### 关联容器添加元素
- `c.insert(v)` or `c.emplace(args)` v是`value_type`类型的对象,args用来构造一个元素。
  对于map和set，只有当元素的key不在c中才出入函数返回一个pair,第一个成员是一个迭代器,指向具有给定关键字的元素,第二个成员是一个bool值,指出元素是否插入成功
  
  对于multimap和multiset,函数总是返回一个迭代器,指向新添加的元素
- `c.insert(b,e)` or `c.insert(il)` il是花括号列表
- `c.insert(p,v)` or `c.emplace(p,args)` 

#### 删除元素
- `c.erase(k)` 删除关键字为k的元素,返回删除的元素数量
- `c.erase(p)` 删除迭代器p指向的元素,返回指向p之后元素的迭代器
- `c.erase(b,e)` 删除迭代器b和e指定的范围内的元素,返回e

#### map的下标操作

map和unordered_map支持下标操作和一个对应的at函数
我们不能对一个multimap或者unordered_multimap执行下标操作,因为一个关键字可能对应对各value

```cpp
    map<string,int> word_count;
    word_count["Anna"] = 1;
```
如果关键字在map中,可以修改对应value,如果关键不在map中,将会插入,并且对应value进行值初始化
上述代码将执行的操作
1. 在word_count中查找关键字为Anna的元素,未发现
2. 创建一个关键字为Anna的元素,并进行值初始化,在本例中值是0
3. 将1赋值给Anna对应的value
```cpp
    map<string,int> word_count;
    word_count.insert({"Anna",1})
    cout << word_count["Anna"] << endl;
```
下标操作当右值,可以返回对应的value

关键字的返回是一个左值,可读可写

#### 访问元素

lower_bound和upper_bound不适用于无序容器
下标和at操作只适用于非const的map和unordered_map
- `c.find(k)` 返回一个迭代器,指向第一个关键字为k的元素,如果k不在c中,则返回尾迭代器
- `c.count(k)` 返回关键字等于k的元素的数量
- `c.lower_bound(k)` 返回一个迭代器,指向第一个关键字>=k的元素
- `c.upper_bound(k)` 返回一个迭代器,指向第一个关键字>k的元素
- `c.equal_range(k)` 返回一个迭代器pair,表示关键字等于k的元素的范围,若k不存在,则pair的两个成员均等于c.end()

### 无序容器

- `c.find(k)` 返回一个迭代器,指向第一个关键字为k的元素,如果k不在c中,则返回尾迭代器
- `c.count(k)` 返回关键字等于k的元素的数量

- `c.insert({k,v})` unordered_map和unordered_multimap支持此操作
- `c.insert(key)` unordered_set支持
  
- `c.erase(k)` 删除关键字为k的元素,返回删除的元素数量
- `c.erase(p)` 删除迭代器p指向的元素,返回指向p之后元素的迭代器
- `c.erase(b,e)` 删除迭代器b和e指定的范围内的元素,返回e
  
支持迭代器遍历和 `:`遍历
```cpp
int main() {
    unordered_map<int,int> m1;
    for (int i = 0; i < 10; i++) {
        m1.insert({i,i*2});
    }

    for (auto it = m1.begin(); it != m1.end(); it++) {
        cout << it->first << " " << it->second << endl;
    }
    for (auto kv:m1) {
        cout << kv.first << " " << kv.second << endl;
    }
}
```

  