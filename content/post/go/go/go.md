---
title: "GO语言相关"
description: 
date: 2023-09-06T15:48:30+08:00
image:
url: /go
math: false
comments: false
draft: true
categories:
    - go
---

[reference1](https://github.com/golang101/golang101)

# 数据结构

## slice and map

切片的内部结构如下
```go
type _slice struct {
    elements unsafe.Pointer //  引用底层存储在间接部分上的元素
    len      int // 长度
    cap      int  // 容量
}
```

### 切片的`append`操作

通过`append`操作之后，新切片和旧切片不一定都在一个内容空间，因此是可能发生复制的，主要和cap有关

当一个切片被用做一个`append`函数调用中的基础切片时
- 如果添加的元素数量大于此基础切片的冗余元素槽位数(cap-len),则一个新的底层内存片段将被开辟出来存放结果切片的元素，这时，基础切片和结果切片不共享任何底层元素
- 否则，不会有底层内存片段被开辟出来，这时，基础切片中所有的元素同时也属于结果切片，两者在同一个内容片段上

### 赋值操作
通过切片的底层数据结构可以看出
```go
a := []int{1,2,3}
b := a
```
上面这个代码片段中,a,b是共享底层内存空间的，因此，对a的修改会影响到b，map也是如此

### 容器元素的可寻址性
- 如果一个数据是可寻址的，则它的元素也是可寻址的，反之亦然。
- 一个切片值的任何元素都是可寻址的
- 任何映射元素都是不可寻址的

### 从数组或者切片中派生子切片

根据切片的数据结构，派生子切片也是共享底层数组的

复制切片,利用内置函数copy
```go
package main

import "fmt"

func main() {
        a := []int{1,2,3,4}
        b := make([]int,5)
        fmt.Println(len(a),a)
        n := copy(b,a)
        fmt.Println(n,len(b),b)
        b[0] = 10
        fmt.Println(a,b)
}
```
使用copy函数之后，两者就是不共享底层数组的

### 遍历容器元素

使用的语法如下
```go
for key, element := range aContainer {

}
```
如果aContainer是切片，那么key是索引，element是元素值
如果aContainer是map，那么key是键，element是值

关于for-range语法有三个注意点
1. 被遍历的aContainer是原来容器的一个副本,而且是一个浅层复制
- 如果`aContainer` 是一个数组,那么在遍历过程中对此数组的修改不会体现在循环中
- 如果`aContainer`是一个切片(或者映射),那么在遍历过程中对此切片(映射)元素的修改将体现到循环变量。原因是此切片(映射)的副本和自己共享底层内存

2. 在遍历过程中，`aContainer`的副本的一个键值元素对将被赋值(赋值)给循环变量。所以对循环变量的直接部分的修改不会体现在`aContainer`中的对应元素中

3. 所有被遍历的简直对将被赋值给同一对循环变量实例

证明实例2:
```go
package main

import "fmt"

func main() {
    type Person struct {
        name string
        age int
    }
    persons := []Person{
        {"Jack", 20},
        {"Lucy", 18},
    }
    for i, p := range persons {
        p.age++ // 此修改不在在原数组中生效
        fmt.Println(i,p)
    }
    fmt.Println(persons)
}
```

利用for-range向切片中添加元素,这种方式是不报错的，因此for-range中遍历的是原切片的一个副本
```go
package main

func main() {
    a := []int{1,2,3}
    for i, ele := range a {
        a = append(a, ele*2)
    }
    fmt.Println(a)
}
```
# 锁

go的锁来自于sync包,为了避免各种异常行为，最好不要复制sync标准库包中提供的类型的值。

## sync.WaitGroup
方法
- `add(delta int)`
- `Done()`
- `Wait()`

这个对象主要实现的是信号量操作
`add`添加数值,`Done`数值减少1,`Wait`阻塞当前携程,直到数值为0
```go
func main() {

    rand.Seed(time.Now().UnixNano())
    
    const N = 5
    var values [N]int32

    var wgA, wgB sync.WaitGroup
    wgA.Add(N)
    wgB.Add(1)
    for i := 0; i < N; i++ {
        go func(i int) {
            wgB.Wait() // 等待wgB
            log.Printf("values[%v]=%v \n",i values[i])
            wgA.Done()
        }(i)
    }
    for i := 0; i < N; i++ {
        values[i] = 50 + rand.Int31n(50)
    }
    wgB.Done() // 发出一个广播通知
    wgA.Wait() // 等待所有的协程完成
}
```
## sync.Once

每个`*sync.Once`值有一个`Do(f func())`方法,此方法传入一个类型为`func()`的函数

通过使用sync.Once包装,保证了该方法如果在多个协程中多次调用,也只会执行一次

一般来说，一个`sync.Once`值被用来确保一段代码在一个并发程序中被执行且仅被执行一次。

```go
package main

import (
	"log"
	"sync"
)

func main() {
	log.SetFlags(0)

	x := 0
	doSomething := func() {
		x++
		log.Println("Hello")
	}
	var wg sync.WaitGroup
	var once sync.Once
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			once.Do(doSomething)
			log.Println("World")
		}()
	}
	wg.Wait()
	log.Println("x =", x) // x = 1
}
// Output
// Hello
// World
// World
// World
// World
// World
// x = 1
```

## sync.Mutex(互斥锁)和sync.RWMutex(读写锁)类型

`*sync.Mutex`和`*sync.RWMutex`类型都实现了`sync.Locker`接口类型。 所以这两个类型都有两个方法：`Lock()`和`Unlock()`，用来保护一份数据不会被多个使用者同时读取和修改。

除了`Lock()`和`Unlock()`这两个方法，`*sync.RWMutex`类型还有两个另外的方法：`RLock()`和`RUnlock()`，用来支持多个读取者并发读取一份数据但防止此份数据被某个数据写入者和其它数据访问者（包括读取者和写入者）同时使用。

**Mutex**

一个`Mutex`值常称为一个互斥锁。 一个`Mutex`零值为一个尚未加锁的互斥锁。 一个（可寻址的）`Mutex`值`m`只有在未加锁状态时才能通过`m.Lock()`方法调用被成功加锁。

换句话说，一旦`m`值被加了锁（亦即某个`m.Lock()`方法调用成功返回）， 一个新的加锁试图将导致当前协程进入阻塞状态，直到此`Mutex`值被解锁为止（通过`m.Unlock()`方法调用）。

**RWMutex**
一个`RWMutex`值常称为一个读写互斥锁，它的内部包含两个锁：一个写锁和一个读锁。 对于一个可寻址的`RWMutex`值`rwm`，数据写入者可以通过方法调用`rwm.Lock()`对rwm加写锁，或者通过`rwm.RLock()`方法调用对`rwm`加读锁。 方法调用`rwm.Unlock()`和`rwm.RUnlock()`用来解开rwm的写锁和读锁。

对于一个可寻址的`RWMutex`值`rwm`，下列规则存在：
- `rwm`的写锁只有在写锁和读锁都未加锁的情况下才能成功，否则阻塞
- 当`rwm`的**写锁正处于加锁状态**的时候，任何新的对之加写锁或者加读锁的操作试图都将导致当前协程进入阻塞状态，直到此写锁被解锁
- 当`rwm`的读锁正处于加锁状态的时候，新的加写锁的操作试图将导致当前协程进入阻塞状态。 但是，一个新的加读锁的操作试图将成功
- 假设`rwm`的读锁正处于加锁状态的时候，为了防止后续数据写入者没有机会成功加写锁，后续发生在某个被阻塞的加写锁操作试图之后的所有加读锁的试图都将被阻塞。
- 假设`rwm`的写锁正处于加锁状态的时候，（至少对于标准编译器来说，）为了防止后续数据读取者没有机会成功加读锁，发生在此写锁下一次被解锁之前的所有加读锁的试图都将在此写锁下一次被解锁之后肯定取得成功，即使所有这些加读锁的试图发生在一些仍被阻塞的加写锁的试图之后。

## sync.Cond

`sync.Cond`类型提供了一种有效的方式来实现多个协程间的通知  

# 通道