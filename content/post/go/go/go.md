---
title: "GO语言基础数据结构"
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

[golang101](https://github.com/golang101/golang101)

# 基本数据类型

Go支持如下内置基本类型
1. 一种内置布尔类型: `bool`
2. 11种内置整数类型: `int8`, `int16`, `int32`, `int64`, `int`, `uint8`, `uint16`, `uint32`, `uint64`, `uint`, `uintptr`
3. 两种内置浮点数类型: `float32`, `float64`
4. 两种内置复数类型: `complex64`, `complex128`
5. 一种内置字符串类型: `string`

这17种内置类型各自属于一种Go的种类(Kind),并且不导出即可使用

内置类型别名
1. `byte`是`uint8`的别名
2. `rune`是`int32`的别名

`uintptr`类型是一种无符号整数类型，用来存放一个指针的值,尺寸依赖具体的编译器实现

**类型零值**

每种类型都有一个零值。一个类型的零值可以看作是此类型的默认值
- `bool`的零值为false
- 数值类型的零值为0
- 字符串类型的零值为空字符串

在Go中，一个rune值表示一个Unicode码点。 一般说来，我们可以将一个Unicode码点看作是一个Unicode字符。 但是，我们也应该知道，有些Unicode字符由多个Unicode码点组成。 每个英文或中文Unicode字符值含有一个Unicode码点。

**值的可寻址性**

在Go中,有些值是可以被寻址的。
所有变量都是可寻址的,所有常量都是不可寻址的

# 组合数据类型
- 指针类型,类C指针
- 结构体类型,类C结构体
- 函数类型,是Go中的一等公民
- 容器类型
  - 数组(array)
  - 切片(slice)
  - 映射(map)
- 通道类型
- 接口类型

每种上面提到的基本类型和组合类型都对应着一个类型种类（kind）。除了这些种类，今后将要介绍的非类型安全指针类型属于另外一个新的类型种类。

底层类型(underlying type)
- 一个内置类型的底层类型为它自己
- `unsafe`标准库包中定义的`Pointer`类型的底层类型为`*T`，其中`T`表示任意一个类型,`unsafe.Pointer`也可以视为一个内置类型
- 一个无名类型（必为一个组合类型）的底层类型为它自己。
- 在一个类型声明中，新声明的类型和源类型共享底层类型。(声明类型别名的时候)

**值(value)**

一个类型的一个实例称为此类型的一个值。一个类型可以有很多不同的值，其中一个为它的零值。 同一类型的不同值共享很多相同的属性。

每个类型有一个零值。一个类型的零值可以看作是此类型的默认值。 预声明的标识符nil可以看作是`slice`、`map`、`函数`、`通道`、`指针（包括非类型安全指针）`和`接口类型`的零值的字面量表示。

**值部**

每个这样的值都有一个直接部分，但是有一些值还可能有一个或多个间接部分。每个值部分在内存中都占据一段连续空间。 通过安全或者非安全指针，一个值的间接部分被此值的直接部分所引用。

在赋值中,底层键值值部不会被复制,只有直接支部会被复制

事实上,对于字符串值和接口值的赋值,上述描述在理论上并百分百正确。 
官方FAQ明确说明了在一个接口值的赋值中，接口的底层动态值将被复制到目标值。

但是，因为一个接口值的动态值是只读的，所以在接口值的赋值中，官方标准编译器并没有复制底层的动态值。这可以被视为是一个编译器优化。 对于字符串值的赋值，道理是一样的。(因为底层只读,就算指向了同一片内存也不会有问题)

## 指针

指针是Go中的一种类型分类（kind）。 一个指针可以存储一个内存地址；从地址通常为另外一个值的地址。

在Go中，一个无名指针类型的字面形式为`*T`，其中`T`为一个任意类型。类型`T`称为指针类型`*T`的基类型（base type）。 如果一个指针类型的基类型为`T`，则我们可以称此指针类型为一个T指针类型。

虽然我们可以声明具名指针类型，但是一般不推荐这么做，因为无名指针类型的可读性更高。

```go
*int // 一个base type is int的无名指针类型
type Ptr *int // Ptr是一个具名指针类型，它的基类型为int。
```

可以通过`new`内置函数来为任何类型的值开辟一块内存空间,并将此内存块的起始地址作为返回

指针的零值为`nil`,指针的零值是一个特殊的指针值，它表示此指针不指向任何一个值

指针解引用

```go
p0 := new(int)
fmt.Println(p0) // address
fmt.Println(*p0) // 0
```

在Go中返回一个局部变量的地址是安全的,因为Go存在逃逸分析,如果一个局部变量的地址被返回,那么这个局部变量就会被分配在堆上

为了安全起见，Go指针在使用上相对于C指针有很多限制。 通过施加这些限制，Go指针保留了C指针的好处，同时也避免了C指针的危险性。
1. Go指针不支持算术运算
2. 一个指针类型的值不能被随意转换为另一个指针类型(要求底层类型一致)
3. 一个指针值不能和其它任一指针类型的值进行比较(要求指针类型相同且其中一个必须是nil)

上述Go指针的限制是可以被打破的

`unsafe`标准库包中提供的非类型安全指针（`unsafe.Pointer`）机制可以被用来打破上述Go指针的安全限制。 `unsafe.Pointer`类型类似于C语言中的`void*`。 但是，通常地，非类型安全指针机制不推荐在Go日常编程中使用。

## array

## slice

切片的内部结构
```go
type _slice struct {
    elements unsafe.Pointer // 引用底层元素
    len int // 当前的元素个数
    cap int // 切片的容量
}
```
切片的`append`方法

可以通过调用内置`append`函数来向一个基础切片添加元素而得到一个新的切片
这个新的结果切片可能和基础切片共享其实元素，也可能不共享，具体取决于切片的容量(以及长度)和添加的元素数量

当一个切片被用做一个`append`函数调用中的基础切片，
- 如果添加的元素数量大于此（基础）切片的冗余元素槽位的数量，则一个新的底层内存片段将被开辟出来并用来存放结果切片的元素。 这时，基础切片和结果切片不共享任何底层元素。
- 否则，不会有底层内存片段被开辟出来。这时，基础切片中的所有元素也同时属于结果切片。两个切片的元素都存放于同一个内存片段上。

切片扩容的原则(Go 1.18)
1. 如果doubleCap(oldCap+oldCap)还不够newLen,则直接返回newLen

下面的情况都是双倍容量足够了
2. 在oldcap < 256 时,直接按照双倍扩容
3. oldcap >= 256时,按照1.25倍扩容,直到大于或等于newLen,这个公式为两者提供了平滑的过渡 `newcap += (newcap + 3*threshold) >> 2`
4. if newcap <= 0, return newLen(规定了初始容量为0时,按照newLen扩容)

在得到了newcap之后,还会使用`rouncupsize(uintptr(newcap)*goarch.PtrSize)`函数来对newcap进行处理,进行内存对齐(分配8的倍数的字节的内存)

```go
// src/runtime/slice.go:155
// growslice allocates new backing store for a slice.
//
// arguments:
//
//	oldPtr = pointer to the slice's backing array
//	newLen = new length (= oldLen + num)
//	oldCap = original slice's capacity.
//	   num = number of elements being added
//	    et = element type
//
// return values:
//
//	newPtr = pointer to the new backing store
//	newLen = same value as the argument
//	newCap = capacity of the new backing store
//
// Requires that uint(newLen) > uint(oldCap).
// Assumes the original slice length is newLen - num
//
// A new backing store is allocated with space for at least newLen elements.
// Existing entries [0, oldLen) are copied over to the new backing store.
// Added entries [oldLen, newLen) are not initialized by growslice
// (although for pointer-containing element types, they are zeroed). They
// must be initialized by the caller.
// Trailing entries [newLen, newCap) are zeroed.
//
// growslice's odd calling convention makes the generated code that calls
// this function simpler. In particular, it accepts and returns the
// new length so that the old length is not live (does not need to be
// spilled/restored) and the new length is returned (also does not need
// to be spilled/restored).
func growslice(oldPtr unsafe.Pointer, newLen, oldCap, num int, et *_type) slice {
	oldLen := newLen - num
	if raceenabled {
		callerpc := getcallerpc()
		racereadrangepc(oldPtr, uintptr(oldLen*int(et.Size_)), callerpc, abi.FuncPCABIInternal(growslice))
	}
	if msanenabled {
		msanread(oldPtr, uintptr(oldLen*int(et.Size_)))
	}
	if asanenabled {
		asanread(oldPtr, uintptr(oldLen*int(et.Size_)))
	}

	if newLen < 0 {
		panic(errorString("growslice: len out of range"))
	}

	if et.Size_ == 0 {
		// append should not create a slice with nil pointer but non-zero len.
		// We assume that append doesn't need to preserve oldPtr in this case.
		return slice{unsafe.Pointer(&zerobase), newLen, newLen}
	}
    // start from this
    // 获取新的容量
	newcap := nextslicecap(newLen, oldCap)

    // 根据新的容量扩容
	var overflow bool
	var lenmem, newlenmem, capmem uintptr
	// Specialize for common values of et.Size.
	// For 1 we don't need any division/multiplication.
	// For goarch.PtrSize, compiler will optimize division/multiplication into a shift by a constant.
	// For powers of 2, use a variable shift.
	noscan := !et.Pointers()
	switch {
	case et.Size_ == 1:
		lenmem = uintptr(oldLen)
		newlenmem = uintptr(newLen)
		capmem = roundupsize(uintptr(newcap), noscan)
		overflow = uintptr(newcap) > maxAlloc
		newcap = int(capmem)
	case et.Size_ == goarch.PtrSize:
		lenmem = uintptr(oldLen) * goarch.PtrSize
		newlenmem = uintptr(newLen) * goarch.PtrSize
		capmem = roundupsize(uintptr(newcap)*goarch.PtrSize, noscan)
		overflow = uintptr(newcap) > maxAlloc/goarch.PtrSize
		newcap = int(capmem / goarch.PtrSize)
	case isPowerOfTwo(et.Size_):
		var shift uintptr
		if goarch.PtrSize == 8 {
			// Mask shift for better code generation.
			shift = uintptr(sys.TrailingZeros64(uint64(et.Size_))) & 63
		} else {
			shift = uintptr(sys.TrailingZeros32(uint32(et.Size_))) & 31
		}
		lenmem = uintptr(oldLen) << shift
		newlenmem = uintptr(newLen) << shift
		capmem = roundupsize(uintptr(newcap)<<shift, noscan)
		overflow = uintptr(newcap) > (maxAlloc >> shift)
		newcap = int(capmem >> shift)
		capmem = uintptr(newcap) << shift
	default:
		lenmem = uintptr(oldLen) * et.Size_
		newlenmem = uintptr(newLen) * et.Size_
		capmem, overflow = math.MulUintptr(et.Size_, uintptr(newcap))
		capmem = roundupsize(capmem, noscan)
		newcap = int(capmem / et.Size_)
		capmem = uintptr(newcap) * et.Size_
	}

	// The check of overflow in addition to capmem > maxAlloc is needed
	// to prevent an overflow which can be used to trigger a segfault
	// on 32bit architectures with this example program:
	//
	// type T [1<<27 + 1]int64
	//
	// var d T
	// var s []T
	//
	// func main() {
	//   s = append(s, d, d, d, d)
	//   print(len(s), "\n")
	// }
	if overflow || capmem > maxAlloc {
		panic(errorString("growslice: len out of range"))
	}

	var p unsafe.Pointer
	if !et.Pointers() {
		p = mallocgc(capmem, nil, false)
		// The append() that calls growslice is going to overwrite from oldLen to newLen.
		// Only clear the part that will not be overwritten.
		// The reflect_growslice() that calls growslice will manually clear
		// the region not cleared here.
		memclrNoHeapPointers(add(p, newlenmem), capmem-newlenmem)
	} else {
		// Note: can't use rawmem (which avoids zeroing of memory), because then GC can scan uninitialized memory.
		p = mallocgc(capmem, et, true)
		if lenmem > 0 && writeBarrier.enabled {
			// Only shade the pointers in oldPtr since we know the destination slice p
			// only contains nil pointers because it has been cleared during alloc.
			//
			// It's safe to pass a type to this function as an optimization because
			// from and to only ever refer to memory representing whole values of
			// type et. See the comment on bulkBarrierPreWrite.
			bulkBarrierPreWriteSrcOnly(uintptr(p), uintptr(oldPtr), lenmem-et.Size_+et.PtrBytes, et)
		}
	}
	memmove(p, oldPtr, lenmem)

	return slice{p, newLen, newcap}
}

//
func nextslicecap(newLen, oldCap int) int {
	newcap := oldCap
	doublecap := newcap + newcap
	if newLen > doublecap {
		return newLen
	}

	const threshold = 256
	if oldCap < threshold {
		return doublecap
	}
	for {
		// Transition from growing 2x for small slices
		// to growing 1.25x for large slices. This formula
		// gives a smooth-ish transition between the two.
		newcap += (newcap + 3*threshold) >> 2

		// We need to check `newcap >= newLen` and whether `newcap` overflowed.
		// newLen is guaranteed to be larger than zero, hence
		// when newcap overflows then `uint(newcap) > uint(newLen)`.
		// This allows to check for both with the same comparison.
		if uint(newcap) >= uint(newLen) {
			break
		}
	}

	// Set newcap to the requested cap when
	// the newcap calculation overflowed.
	if newcap <= 0 {
		return newLen
	}
	return newcap
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

## map
```go
type _map *hashtableImpl
```

## channel

[golang channel 最详细的源码剖析](https://zhuanlan.zhihu.com/p/297053654)

```go
type _channel *channelImpl

// 类型为T的双向通道
chan T 
// 只发送 
chan<- T
// 只接收
<-chan T
// chan 零值为nil
// 使用make初始化,有缓冲
ch := make(chan int, 10)
// 无缓冲
ch := make(chan int)
```
本质上来说,channel就是一个环形队列(ringbuffer)的实现

通过make声明通道

```go 
// runtime/chan.go
func makechan(t *chantype, size int) *hchan {
	// 参数校验
	// 初始化hchan结构
}

type hchan strcut {
	qcount uint // queue里有效元素的个数,在出队,入队的时候改变
	dataqsiz uint // 初始化的时候赋值,之后不在改变,指名了数组buffer的大小
	buf unsafe.Pointer // 指向buffer数组的地址
	elemsize uint16 // 元素的大小,和 dataqsiz 配合使用就能知道 buffer 内存块的大小了;
	closeed uint32 // 通道是否关闭
	elemtype *_type // 元素的类型
	sendx uint // send index
	recvx uint // receive index
	recvq waitq // receive wait queue,抽象成 waiters
	sendq waitq // send wait queue,抽象成 waiters
	// 互斥资源的保护锁，官方特意说明，在持有本互斥锁的时候，绝对不要修改 Goroutine 的状态，不能很有可能在栈扩缩容的时候，出现死锁
	lock mutex 
}

// waitq 是一个等待队列，用于管理等待发送操作或接收操作的 goroutine。
// 它的作用是在通道的发送和接收操作中提供阻塞和唤醒的机制。
type waitq struct {
	first *sudog
	last *sudog
}
```

通道操作A,协程R发送数据到未关闭的通道 => `ch <- 1`
```go
// runtime.chansend
// 返回结果是是否发送成功
// c chan数据结构
// ep 指向发送的数据
// block:指明该发送是否是block模式,如果是,则发送的时候不能发送则会阻塞
// callerpc
// ret: 返回是否发送成功

// nil 通道: !block => false, block => 永久阻塞
// c.recvq 和 buf之间的关系:如果c.recvq有等待的goroutine,buf一定为空
// if c.recvq 中有元素,则直接发送给对应的协程
// else if c.qcount < c.dataqsiz(缓冲区未满), 则将数据缓存到队列中
// else {
// 		if !block => return false
// 		else 当前协程和对应的数据加入到发送等待队列中(c.sendq),阻塞当前协程
// }
func chansend(c *hchan, ep unsafe.Pointer, block bool, callerpc uintptr) bool {
	if c == nil {
		//如果非堵塞模式，则直接返回false
		if !block {
			return false
		}
		// nil channel 发送数据会永远阻塞下去
		// 挂起当前 goroutine
		gopark(nil, nil, waitReasonChanSendNilChan, traceEvGoStop, 2)
		throw("unreachable")
	}

	// 快速检测模式
	// 非阻塞模式,通道未关闭且buff满
	if !block && c.closed == 0 && full(c) {
		return false
	}
	
	var t0 int64
	//未启用阻塞分析，由于CPU分支预测
	if blockprofilerate > 0 {
		t0 = cputicks()
	}

	//上锁
	lock(&c.lock)

	//chan已经关闭，解锁，panic
	if c.closed != 0 {
		unlock(&c.lock)
		panic(plainError("send on closed channel"))
	}
	// 如果在接收等待队列上存在正在等待的G，则直接将数据发送
	// 不必将数据缓存到队列中
	if sg := c.recvq.dequeue(); sg != nil {
		send(c, sg, ep, func() { unlock(&c.lock) }, 3)
		return true
	}
	/**
		如果当前chan的缓存区未满，将数据缓存到队列中；否则阻塞当前G
	 */

	//当前chan的缓存区未满
	if c.qcount < c.dataqsiz {
		//计算下一个缓存区位置指针
		qp := chanbuf(c, c.sendx)
		//将数据保存到缓冲区队列
		typedmemmove(c.elemtype, qp, ep)
		//sendx位置往后移动一位
		c.sendx++
		//如果c.sendx == c.dataqsiz，表示sendx索引已经达到缓冲队列最尾部了，则将sendx移动到0（第一个位置），这个是环形队列思维
		if c.sendx == c.dataqsiz {
			c.sendx = 0
		}
		//Chan中的元素个数+1
		c.qcount++
		//解锁，返回即可
		unlock(&c.lock)
		return true
	}

	//如果非堵塞模式，缓冲区满了则直接解锁，返回false
	if !block {
		unlock(&c.lock)
		return false
	}
	//缓冲队列已满或者创建的不带缓冲的channel，则阻塞当前G
	//获取当前goroutine
	gp := getg()
	// 获取一个sudog对象并设置其字段
	mysg := acquireSudog()
	mysg.releasetime = 0
	if t0 != 0 {
		mysg.releasetime = -1
	}

	mysg.elem = ep //将指向发送数据的指针保存到 elem 中
	mysg.waitlink = nil
	mysg.g = gp //将g指向当前的goroutine
	mysg.isSelect = false
	mysg.c = c //当前阻塞的 channel
	gp.waiting = mysg
	gp.param = nil  // param 可以用来传递数据，其他 goroutine 唤醒该 goroutine 时可以设置该字段，然后根据该字段做一些判断
	c.sendq.enqueue(mysg)// 将sudog加入到channel的发送等待队列hchan.sendq中

	atomic.Store8(&gp.parkingOnChan, 1)
	// 当前 Goroutine 切换为等待状态并阻塞等待其他的Goroutine从 channel 接收数据并将其唤醒
	gopark(chanparkcommit, unsafe.Pointer(&c.lock), waitReasonChanSend, traceEvGoBlockSend, 2)

	// 在没有其他的接收队列将数据复制到队列中时候，需要保证当前需要被发送的的值一直是可用状态
	KeepAlive(ep)
	/**
		协程被唤醒后
	 */
	if mysg != gp.waiting {
		throw("G waiting list is corrupted")
	}
	//更新goroutine相关的对象信息
	gp.waiting = nil
	gp.activeStackChans = false
	closed := !mysg.success
	gp.param = nil
	if mysg.releasetime > 0 {
		blockevent(mysg.releasetime-t0, 2)
	}
	mysg.c = nil
	//释放sudog对象
	releaseSudog(mysg)
	//如果channel已经关闭
	if closed {
		// close标志位为0，则抛出假性唤醒异常
		if c.closed == 0 {
			throw("chansend: spurious wakeup")
		}
		//直接panic
		panic(plainError("send on closed channel"))
	}
	return true
}

// 通道操作B: 协程R从未关闭的通道中接受数据 => `v := <-ch`
//go 1.20.3 path: /src/runtime/chan.go
// chanrecv receives on channel c and writes the received data to ep.
// ep may be nil, in which case received data is ignored.
// If block == false and no elements are available, returns (false, false).
// Otherwise, if c is closed, zeros *ep and returns (true, false).
// Otherwise, fills in *ep with an element and returns (true, true).
// A non-nil ep must point to the heap or the caller's stack.
// selected 指明在select语句中是否可以立刻返回(非阻塞)
// received 指明是否接收到数据

// sendq和buf的关系,如果sendq有等待的goroutine, 则buf一定已经满了,而且没有recv的协程
func chanrecv(c *hchan, ep unsafe.Pointer, block bool) (selected, received bool) {

	if c == nil {
		//如果chan为空且是非阻塞调用，那么直接返回 (false,false)
		if !block {
			return
		}
		// 阻塞调用直接等待
		gopark(nil, nil, waitReasonChanReceiveNilChan, traceEvGoStop, 2)
		throw("unreachable")
	}

	/**
	快速检测，在非阻塞模式下，和发送一样有些条件不需要加锁就可以直接判断返回。

	当前非阻塞并且chan未关闭，并符合下列条件之一：
	1. 非缓冲channel且没有待发送者
	2. 缓冲channel且是缓冲区为空
	*/
	if !block && empty(c) {
		//chan未关闭，直接返回(false,false)
		if atomic.Load(&c.closed) == 0 {
			return
		}
		//channel 处于关闭，并且empty(c)，返回(true,false)
		if empty(c) {
			if ep != nil {
				//将接收的值置为空值
				typedmemclr(c.elemtype, ep)
			}
			return true, false
		}
	}

	//未启用阻塞分析，由于CPU分支预测
	var t0 int64
	if blockprofilerate > 0 {
		t0 = cputicks()
	}

	//加锁
	lock(&c.lock)
  
  //channel 处于关闭
	if c.closed != 0 {
    //如果channel元素为空
		if c.qcount == 0 {
      		//如果竞态检测功能已启用（即 raceenabled 为 true），则调用 raceacquire() 函数检测
			if raceenabled {
				raceacquire(c.raceaddr())
			}
      		//解锁
			unlock(&c.lock)
			if ep != nil {
        	//将接收的值置为空值
				typedmemclr(c.elemtype, ep)
			}
			return true, false
		}
	} else {
    //待发送队列sendq中有 goroutine，说明是非缓冲channel或者缓冲已满的 channel,将数据从待发送者复制给接收者
		if sg := c.sendq.dequeue(); sg != nil {
			recv(c, sg, ep, func() { unlock(&c.lock) }, 3)
			return true, true
		}
	}

	//chan的缓存队列中还有数据
	if c.qcount > 0 {
		//获取一个缓存队列数据的指针地址
		qp := chanbuf(c, c.recvx)
		if ep != nil {
			//将该数据复制到接收对象
			typedmemmove(c.elemtype, ep, qp)
		}
		//清空该指针地址的数据
		typedmemclr(c.elemtype, qp)
		//recvx+1
		c.recvx++
		//如果接收游标 等于环形链表的值，则接收游标清零。
		if c.recvx == c.dataqsiz {
			c.recvx = 0
		}
		//循环数组buf元素数量-1
		c.qcount--
		unlock(&c.lock)
		return true, true
	}

	//非阻塞接收，因为chan的缓存中没有数据，则解锁，selected 返回 false，因为没有接收到值
	if !block {
		unlock(&c.lock)
		return false, false
	}

	// 缓冲区队列没有数据可以读取，则将当前G打包成Sudo结构并加入到接收等待队列
	gp := getg()
  
	/**
	创建一个sudog结构体，并将其与当前的goroutine (gp) 关联。
	sudog结构体用于在并发环境中进行同步操作和调度。其中的字段和赋值操作可能会在其他代码中使用
	*/
  
  //创建一个新的sudog结构体，并将其赋值给变量mysg
	mysg := acquireSudog()
	mysg.releasetime = 0
	if t0 != 0 {
		mysg.releasetime = -1
	}

	mysg.elem = ep
	mysg.waitlink = nil
	gp.waiting = mysg
	mysg.g = gp
	mysg.isSelect = false
	mysg.c = c
	gp.param = nil
	c.recvq.enqueue(mysg) // 加入到接收等待队列recvq中

	atomic.Store8(&gp.parkingOnChan, 1)
	// 阻塞等待被唤醒
	gopark(chanparkcommit, unsafe.Pointer(&c.lock), waitReasonChanReceive, traceEvGoBlockRecv, 2)

	if mysg != gp.waiting {
		throw("G waiting list is corrupted")
	}
	//唤醒后，设置goroutine的部分字段值，并释放该g的Sudo
	gp.waiting = nil
	gp.activeStackChans = false
	if mysg.releasetime > 0 {
		blockevent(mysg.releasetime-t0, 2)
	}
	success := mysg.success
	gp.param = nil
	mysg.c = nil
	releaseSudog(mysg)
	return true, success
}
```

`chansend`函数和`chanrecv`函数的`block`指名当前收发的模式,block = true表示阻塞模式,block = false表示非阻塞模式,如果block = false,那么如果不能收发则会直接返回，否则会阻塞当前goroutine

`select`操作的实现
[Go 语言设计与实现-select](https://draveness.me/golang/docs/part2-foundation/ch05-keyword/golang-select/)
1. `select` 能够在chan上实现非阻塞的收发
2. `select` 在遇到多个chan同时响应时,会随机执行一种情况

在通常情况下，`select` 语句会阻塞当前 Goroutine 并等待多个 Channel 中的一个达到可以收发的状态。但是如果 select 控制结构中包含 `default` 语句，那么这个 `select` 语句在执行时会遇到以下两种情况：
1. 当存在可以收发的 Channel 时，直接处理该 Channel 对应的 `case`；
2. 当不存在可以收发的 Channel 时，执行 `default` 中的语句；

数据结构
```go
type scase struct {
	c *hchan // chan
	elem unsafe.Pointer // data emelent
}
```

编译器在中间代码生成期间会根据 `select` 中 `case` 的不同对控制语句进行优化，这一过程都发生在 `cmd/compile/internal/gc.walkselectcases` 函数中，我们在这里会分四种情况介绍处理的过程和结果：
1. `select` 不存在任何的 `case`； 优化成`block`,直接阻塞
2. `select` 只存在一个 `case`； 改成单一chan
3. `select` 存在两个 `case`，其中一个 `case` 是 `default`；
4. `select` 存在多个 `case`；

```go
// 2
// 单一管道
// 改写前
select {
case v, ok <-ch: // case ch <- v
    ...    
}

// 改写后
if ch == nil {
    block()
}
v, ok := <-ch // case ch <- v
...

// 3
// 非阻塞发送
select {
case ch <- i:
    ...
default:
    ...
}

if selectnbsend(ch, i) {
    ...
} else {
    ...
}

// selectnbsend
func selectnbsend(c *hchan, elem unsafe.Pointer) (selected bool) {
	return chansend(c, elem, false, getcallerpc())
}
// 非阻塞接受
// 改写前
select {
case v <- ch: // case v, ok <- ch:
    ......
default:
    ......
}

// 改写后
if selectnbrecv(&v, ch) { // if selectnbrecv2(&v, &ok, ch) {
    ...
} else {
    ...
}

func selectnbrecv(elem unsafe.Pointer, c *hchan) (selected bool) {
	selected, _ = chanrecv(c, elem, false)
	return
}

func selectnbrecv2(elem unsafe.Pointer, received *bool, c *hchan) (selected bool) {
	selected, *received = chanrecv(c, elem, false)
	return
}
```
常见流程

在默认的情况下，编译器会使用如下的流程处理 select 语句：
1. 将所有`case`转换为包含`chan`以及类型的`runtime.scase`结构体
2. 调用`runtime.selectgo`从多个准备就绪的`chan`中选择一个可执行`runtime.scase`结构体
3. 通过`for`循环生成一组`if`语句,在语句中判断自己是不是被选中的`case`

```go
// 加入存在三个case,那么改写的代码类似余下
selv := [3]scase{}
order := [6]uint16
for i, cas := range cases {
    c := scase{}
    c.kind = ...
    c.elem = ...
    c.c = ...
}
chosen, revcOK := selectgo(selv, order, 3)
if chosen == 0 {
    ...
    break
}
if chosen == 1 {
    ...
    break
}
if chosen == 2 {
    ...
    break
}
```

分析`selectgo`函数的实现
1. 执行一下必要的初始化操作并确定`case`的处理顺序
2. 在循环中根据`case`类型做出不同的处理

c初始化
轮询顺序 pollOrder 和加锁顺序 lockOrder 分别是通过以下的方式确认的：

轮询顺序：通过 `runtime.fastrandn` 函数引入随机性；
加锁顺序：按照 Channel 的地址排序后确定加锁顺序；
随机的轮询顺序可以避免 Channel 的饥饿问题，保证公平性；而根据 Channel 的地址顺序确定加锁顺序能够避免死锁的发生。这段代码最后调用的 runtime.sellock 会按照之前生成的加锁顺序锁定 select 语句中包含所有的 Channel。

循环 #
当我们为 select 语句锁定了所有 Channel 之后就会进入 runtime.selectgo 函数的主循环，它会分三个阶段查找或者等待某个 Channel 准备就绪：

查找是否已经存在准备就绪的 Channel，即可以执行收发操作；
将当前 Goroutine 加入 Channel 对应的收发队列上并等待其他 Goroutine 的唤醒；
当前 Goroutine 被唤醒之后找到满足条件的 Channel 并进行处理；

# function
```go
type _function *functioImpl
```

# string
```go
type _string struct {
    elements *byte // 引用底层存储字符串的字节数组
    len int // 字符串的长度
}
```

# struct


# interface

空接口

```go
type _inferface struct {
    dynamicType *_type // 引用着接口值的动态类型
    dynamicValue unsafe.Pointer // 引用着接口值的动态值
}
```

非空接口
```go
type _interface struct {
    dynamicTypeInfo *struct {
        dynamicType *_type // 引用着接口值的动态类型
        methods []*_function // 引用着动态类型的对应方法列表
    }
    dynamicValue unsafe.Pointer // 引用着接口值的动态值
}
```

# 泛型

# 同步原语

`sync.Once`,`sync.WaitGroup`,`sync.Mutex`,`sync.RWMutex`

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

# 调度器(GMP scheduler)

## 数据结构

### G 

Goroutine 是 Go 语言调度器中待执行的任务，它在运行时调度器中的地位与线程在操作系统中差不多，但是它占用了更小的内存空间，也降低了上下文切换的开销。

Goroutine 在 Go 语言运行时使用私有结构体 runtime.g 表示。这个私有结构体非常复杂，总共包含 40 多个用于表示各种状态的成员变量，这里也不会介绍所有的字段，仅会挑选其中的一部分，首先是与栈相关的两个字段：

```go
// runtime.g
// stack相关
type g struct {
	...
	stack stack
	stackguard0 uintptr
	...
}

type g struct {
	preempt bool // 抢占信号
	preemptStop bool // 抢占时,将状态修改为`_Gpreempted`
	preemptShrink bool // 在同步安全点收缩栈
}

type g struct {
	_panic       *_panic // 最内侧的 panic 结构体
	_defer       *_defer // 最内侧的延迟函数结构体
}

type g struct {
	m              *m // 当前 Goroutine 占用的线程，可能为空；
	sched          gobuf // 存储 Goroutine 的调度相关的数据；
	atomicstatus   uint32 // Goroutine 的状态；
	goid           int64 // Goroutine 的 ID，该字段对开发者不可见
}

// 这些内容会在调度器保存或者恢复上下文的时候用到，其中的栈指针和程序计数器会用来存储或者恢复寄存器中的值，改变程序即将执行的代码。
type gobuf struct {
	sp   uintptr // 栈指针
	pc   uintptr // 程序计数器
	g    guintptr // 持有 runtime.gobuf 的 Goroutine；
	ret  sys.Uintreg // 系统调用的返回值；
	...
}
```
`atomicstatus`字段存储当当前goroutine的状态,有以下9种状态

1. `_Gidle`：刚刚被分配并且还没有被初始化
2. `_Grunnable`：没有执行代码，没有栈的所有权，存储在运行队列中
3. `_Grunning`：可以执行代码，拥有栈的所有权，被赋予了内核线程 M 和处理器 P
4. `_Gsyscall`: 正在执行系统调用，拥有栈的所有权，没有执行用户代码，被赋予了内核线程 M 但是不在运行队列上
5. `_Gwaiting`: 由于运行时而被阻塞，没有执行用户代码并且不在运行队列上，但是可能存在于 Channel 的等待队列上
6. `_Gdead`：没有被使用，没有执行代码，可能有分配的栈
7. `_Gcopystack`: 栈正在被拷贝，没有执行代码，不在运行队列上
8. `_Gpreempted`: 由于抢占而被阻塞，没有执行用户代码并且不在运行队列上，等待唤醒
9. `_Gscan`: 正在被 GC 扫描，没有执行代码，可以与其他状态同时存在

虽然 Goroutine 在运行时中定义的状态非常多而且复杂，但是我们可以将这些不同的状态聚合成三种：**等待中**、**可运行**、**运行中**，运行期间会在这三种状态来回切换：

- 等待中：Goroutine 正在等待某些条件满足，例如 `_Gwaiting`、`_Gsyscall`、`_Gpreempted` 等状态
- 可运行：Goroutine 已经准备就绪，可以在线程运行, `_Grunnable`状态
- 运行中：Goroutine 正在某个线程上运行，即 `_Grunning`；

### M
Go 语言并发模型中的 M 是操作系统线程。调度器最多可以创建 10000 个线程，但是其中大多数的线程都不会执行用户代码（可能陷入系统调用），最多只会有 `GOMAXPROCS` 个活跃线程能够正常运行。

在默认情况下，一个四核机器会创建四个活跃的操作系统线程，每一个线程都对应一个运行时中的 `runtime.m` 结构体。

在大多数情况下，我们都会使用 Go 的默认设置，也就是线程数等于 CPU 数，默认的设置不会频繁触发操作系统的线程调度和上下文切换，所有的调度都会发生在用户态，由 Go 语言调度器触发，能够减少很多额外开销。

`runtime.m`表示操作系统线程,这个结构体字段也很多,只列出一部分感兴趣的
```go
type m struct {
	g0 *g
	curg *g
}
```
其中 `g0` 是持有调度栈的 Goroutine，`curg` 是在当前线程上运行的用户 Goroutine，这也是操作系统线程唯一关心的两个 Goroutine。

g0是一个运行时比较特殊的Goroutine，它会深度参与运行时的调度过程，包括 Goroutine 的创建、大内存分配和 CGO 函数的执行。

```go
// 和p相关的字段
type m struct {
	p puintptr // 正在运行代码的处理器p
	nextp puintptr //暂存的处理器 nextp
	oldp puintptr // 执行系统调用之前使用线程的处理器p
}
```

### P

调度器中的处理器 P 是线程和 Goroutine 的中间层，它能提供线程需要的上下文环境，也会负责调度线程上的等待队列，通过处理器 P 的调度，每一个内核线程都能够执行多个 Goroutine，它能在 Goroutine 进行一些 I/O 操作时及时让出计算资源，提高线程的利用率。

因为调度器在启动时就会创建 `GOMAXPROCS` 个处理器，所以 Go 语言程序的处理器(P)数量一定会等于 `GOMAXPROCS`，这些处理器会绑定到不同的内核线程上。

`runtime.p` 是处理器的运行时表示，作为调度器的内部实现，它包含的字段也非常多,这里我们只关注需要的字段
```go
type p struct {
	m mintptr // 当前绑定的线程
	runqhead uint32 // 队列头
	runqtail uint32 // 队列尾
	runq [256]guintptr // 运行队列
	runnext guintptr // 下一个要运行的Goroutine

	status uint32 // 处理器的状态
}
```

runtime.p 结构体中的状态 `status` 字段会是以下五种中的一种：
1. `_Pidle`：处理器没有运行用户代码或者调度器，被空闲队列或者改变其状态的结构持有，运行队列为空
2. `_Prunning`：被线程M持有,并且正在执行用户代码或者调度器
3. `_Psyscall`：没有执行用户代码，当前线程陷入系统调用
4. `_Pgcstop`：被线程 M 持有，当前处理器由于垃圾回收被停止
5. `_Pdead`: 当前处理器已经不被使用

## 调度器启动

调度器的启动过程是我们平时比较难以接触的过程，不过作为程序启动前的准备工作，理解调度器的启动过程对我们理解调度器的实现原理很有帮助，`runtime`通过 `runtime.schedinit` 初始化调度器：
```go
func schedinit() {
	_g_ := getg()
	...

	sched.maxmcount = 10000 // go语言能够创建的最大线程数

	...
	sched.lastpoll = uint64(nanotime())
	procs := ncpu
	// 从环境变量中获取 GOMAXPROCS 的值
	if n, ok := atoi32(gogetenv("GOMAXPROCS")); ok && n > 0 {
		procs = n
	}
	// procresize 会根据 procs 的值更新 P 的数量
	if procresize(procs) != nil {
		throw("unknown runnable goroutine during bootstrap")
	}
}
```

`runtime.procresize` 更新程序中处理器的数量，在这时整个程序不会执行任何用户 Goroutine，调度器也会进入锁定状态，`runtime.procresize` 的执行过程

1. 如果全局变量 `allp` 切片中的处理器数量少于期望数量，会对切片进行扩容；
2. 使用 `new` 创建新的处理器结构体并调用 `runtime.p.init` 初始化刚刚扩容的处理器；
3. 通过指针将线程 `m0` 和处理器 `allp[0]` 绑定到一起；
4. 调用 `runtime.p.destroy` 释放不再使用的处理器结构；
5. 通过截断改变全局变量 `allp` 的长度保证与期望处理器数量相等；
6. 将除 `allp[0]` 之外的处理器 `P` 全部设置成 `_Pidle` 并加入到全局的空闲队列中

调用 `runtime.procresize` 是调度器启动的最后一步，在这一步过后调度器会完成相应数量处理器的启动，等待用户创建运行新的 Goroutine 并为 Goroutine 调度处理器资源。

### 创建goroutine

通过go关键字创建goroutine, 编译器会把go关键字会转为`runtime.newproc`函数调用,


```go
// 入参是参数大小和表示函数的指针`funcval`
// 它会获取 Goroutine 以及调用方的程序计数器(pc)
// 然后调用`newproc1`函数获取一个新的g结构体,并将其加入到当前P的运行队列中
// 等待满足条件时调用`runtime.wakep`函数唤醒新的处理器执行Goroutine

 func newproc(siz int32, fn *funcval) {
	argp := add(unsafe.Pointer(&fn), sys.PtrSize)
	gp := getg()
	pc := getcallerpc()
	systemstack(func() {
		newg := newproc1(fn, argp, siz, gp, pc)

		_p_ := getg().m.p.ptr()
		runqput(_p_, newg, true)

		if mainStarted {
			wakep()
		}
	})
}
```

`runtime.newproc1` 会根据传入参数初始化一个 g 结构体，我们可以将该函数分成以下几个部分介绍它的实现：
1. 获取或者创建新的 Goroutine 结构体；
2. 将传入的参数移动到 Goroutine 的栈上；
3. 更新Goroutine调度相关的属性
```go
func newproc1(fn *funcval, argp unsafe.Pointer, narg int32, callergp *g, callerpc uintptr) *g {
	//  Goroutine 结构体的创建过程
	// 代码会先从处理器的 gFree 列表中查找空闲的 Goroutine，(gfget(__p__))
	// 如果不存在空闲的 Goroutine，会通过 runtime.malg 创建一个栈大小足够的新结构体。
	_g_ := getg()
	siz := narg
	siz = (siz + 7) &^ 7

	_p_ := _g_.m.p.ptr()
	newg := gfget(_p_)
	if newg == nil {
		newg = malg(_StackMin)
		casgstatus(newg, _Gidle, _Gdead)
		allgadd(newg)
	}
	...

	// 接下来，我们会调用 runtime.memmove 将 fn 函数的所有参数拷贝到栈上，
	// argp 和 narg 分别是参数的内存空间和大小，
	// 我们在该方法中会将参数对应的内存空间整块拷贝到栈上：
	...
	totalSize := 4*sys.RegSize + uintptr(siz) + sys.MinFrameSize
	totalSize += -totalSize & (sys.SpAlign - 1)
	sp := newg.stack.hi - totalSize
	spArg := sp
	if narg > 0 {
		memmove(unsafe.Pointer(spArg), argp, uintptr(narg))
	}
	...
	// 拷贝了栈上的参数之后，runtime.newproc1 会设置新的 Goroutine 结构体的参数，
	// 包括栈指针、程序计数器并更新其状态到 _Grunnable 并返回：
	...
	memclrNoHeapPointers(unsafe.Pointer(&newg.sched), unsafe.Sizeof(newg.sched))
	newg.sched.sp = sp
	newg.stktopsp = sp
	// 设置sched调度信息
	newg.sched.pc = funcPC(goexit) + sys.PCQuantum
	newg.sched.g = guintptr(unsafe.Pointer(newg))
	gostartcallfn(&newg.sched, fn)
	//
	newg.gopc = callerpc
	newg.startpc = fn.fn
	casgstatus(newg, _Gdead, _Grunnable)
	newg.goid = int64(_p_.goidcache)  
	_p_.goidcache++  
	return newg
}
```

获取g结构体有两种方式
1. `gfget` 从当前所在处理器的`gFree`列表或者调度器的`sched.gFree`列表中获取`runtime.g`
2. 调用 `runtime.malg` 创建一个新的`runtime.g`结构体，并将其追加到全局的`allg`列表中

runtime.gfget 中包含两部分逻辑，它会根据处理器中 `gFree` 列表中 Goroutine 的数量做出不同的决策：

1. 当处理器的 Goroutine 列表为空时，会将调度器持有的空闲 Goroutine 转移到当前处理器上，直到 gFree 列表中的 Goroutine 数量达到 32；
2. 当处理器的 Goroutine 数量充足时，会从列表头部返回一个新的 Goroutine；

```go
func gfget(_p_ *p) *g {
retry:
	if _p_.gFree.empty() && (!sched.gFree.stack.empty() || !sched.gFree.noStack.empty()) {
		for _p_.gFree.n < 32 {
			gp := sched.gFree.stack.pop()
			if gp == nil {
				gp = sched.gFree.noStack.pop()
				if gp == nil {
					break
				}
			}
			_p_.gFree.push(gp)
		}
		goto retry
	}
	gp := _p_.gFree.pop()
	if gp == nil {
		return nil
	}
	return gp
}
```
当调度器的 gFree 和处理器的 gFree 列表都不存在结构体时，运行时会调用 `runtime.malg` 初始化新的 `runtime.g` 结构，如果申请的堆栈大小大于 0，这里会通过 `runtime.stackalloc` 分配 2KB 的栈空间：
```go
func malg(stacksize int32) *g {
	newg := new(g)
	if stacksize >= 0 {
		stacksize = round2(_StackSystem + stacksize)
		newg.stack = stackalloc(uint32(stacksize))
		newg.stackguard0 = newg.stack.lo + _StackGuard
		newg.stackguard1 = ^uintptr(0)
	}
	return newg
}
```

**runq**
`runtime.runqput`会将goroutine加入运行队列上,这既可能是全局的运行队列，也可能是处理器本地的运行队列：

```go
func runqput(_p_ *p, gp *g, next bool) {
	if next {
	retryNext:
		oldnext := _p_.runnext
		if !_p_.runnext.cas(oldnext, guintptr(unsafe.Pointer(gp))) {
			goto retryNext
		}
		if oldnext == 0 {
			return
		}
		gp = oldnext.ptr()
	}
retry:
	h := atomic.LoadAcq(&_p_.runqhead)
	t := _p_.runqtail
	if t-h < uint32(len(_p_.runq)) {
		_p_.runq[t%uint32(len(_p_.runq))].set(gp)
		atomic.StoreRel(&_p_.runqtail, t+1)
		return
	}
	if runqputslow(_p_, gp, h, t) {
		return
	}
	goto retry
}
```

1. 当 `next` 为 true 时，将 Goroutine 设置到处理器的 `runnext` 作为下一个处理器执行的任务；
当 next 为 `false` 并且本地运行队列还有剩余空间时，将 Goroutine 加入处理器持有的本地运行队列；
当处理器的本地运行队列已经没有剩余空间时就会把本地队列中的一部分 Goroutine 和待加入的 Goroutine 通过 `runtime.runqputslow` 添加到调度器持有的全局运行队列上；

处理器本地的运行队列是一个使用数组构成的环形链表，它最多可以存储 256 个待执行任务。

## 调度循环

调度器启动之后，Go 语言运行时会调用 `runtime.mstart` 以及 `runtime.mstart1`，前者会初始化 `g0` 的 `stackguard0` 和 `stackguard1` 字段，后者会初始化线程并调用 `runtime.schedule` 进入调度循环：
```go
func schedule() {
	_g_ := getg()

top:
	var gp *g
	var inheritTime bool

	if gp == nil {
		if _g_.m.p.ptr().schedtick%61 == 0 && sched.runqsize > 0 {
			lock(&sched.lock)
			gp = globrunqget(_g_.m.p.ptr(), 1)
			unlock(&sched.lock)
		}
	}
	if gp == nil {
		gp, inheritTime = runqget(_g_.m.p.ptr())
	}
	if gp == nil {
		gp, inheritTime = findrunnable()
	}

	execute(gp, inheritTime)
}
```
1. 为了保证公平，当全局运行队列中有待执行的 Goroutine 时，通过 `schedtick` 保证有一定几率会从全局的运行队列中查找对应的 Goroutine；
2. 从处理器本地的运行队列中查找待执行的 Goroutine；
3. 如果前两种方法都没有找到 Goroutine，会通过 `runtime.findrunnable` 进行阻塞地查找 Goroutine；

`runtime.findrunnable` 的实现非常复杂，这个 300 多行的函数通过以下的过程获取可运行的 Goroutine：

1. 从本地运行队列、全局运行队列中查找；
2. 从网络轮询器中查找是否有 Goroutine 等待运行；
3. 通过 `runtime.runqsteal` 尝试从其他随机的处理器中窃取待运行的 Goroutine，该函数还可能窃取处理器的计时器；
总而言之，当前函数一定会返回一个可执行的 Goroutine，如果当前不存在就会阻塞等待。

接下来由 runtime.execute 执行获取的 Goroutine，做好准备工作后，它会通过 runtime.gogo 将 Goroutine 调度到当前线程上。
```go
func execute(gp *g, inheritTime bool) {
	_g_ := getg()

	_g_.m.curg = gp
	gp.m = _g_.m
	casgstatus(gp, _Grunnable, _Grunning)
	gp.waitsince = 0
	gp.preempt = false
	gp.stackguard0 = gp.stack.lo + _StackGuard
	if !inheritTime {
		_g_.m.p.ptr().schedtick++
	}

	gogo(&gp.sched)
}
```
runtime.gogo 在不同处理器架构上的实现都不同，但是也都大同小异(汇编代码,skip)

它从 `runtime.gobuf` 中取出了 `runtime.goexit` 的程序计数器和待执行函数的程序计数器，其中：

`runtime.goexit` 的程序计数器被放到了栈 `SP` 上；
待执行函数的程序计数器被放到了寄存器 `BX` 上；

需要`runtime.goexit`的原因是我们的函数调用结束之后需要exit协程,这里利用call指令做函数调用的特点,使得函数调用结束后,pc指向`runtime.goexit`函数,从而退出协程

经过一系列复杂的函数调用，我们最终在当前线程的 `g0` 的栈上调用 `runtime.goexit0` 函数，该函数会将 Goroutine 转换会 `_Gdead` 状态、清理其中的字段、移除 Goroutine 和线程的关联并调用 `runtime.gfput` 重新加入处理器的 Goroutine 空闲列表 `gFree`：
```go
TEXT runtime·goexit(SB),NOSPLIT,$0-0
	CALL	runtime·goexit1(SB)

func goexit1() {
	mcall(goexit0)
}

func goexit0(gp *g) {
	_g_ := getg()

	casgstatus(gp, _Grunning, _Gdead)
	gp.m = nil
	...
	gp.param = nil
	gp.labels = nil
	gp.timer = nil

	dropg()
	gfput(_g_.m.p.ptr(), gp)
	schedule()
}
```
在最后`runtime.goexit0`会重新调用`runtime.schedule`触发新一轮的调度

这里介绍的是 Goroutine 正常执行并退出的逻辑，实际情况会复杂得多，多数情况下 Goroutine 在执行的过程中都会经历协作式或者抢占式调度，它会让出线程的使用权等待调度器的唤醒。

## 所有触发调度的时间点

除了上图中可能触发调度的时间点，运行时还会在线程启动 `runtime.mstart` 和 Goroutine 执行结束 `runtime.goexit0` 触发调度。其他的

1. 主动挂起`runtime.gopark -> runtime.park_m` 
2. 系统调用 `runtime.exitsyscall -> runtime.exitsyscall0`
3. 协作式调度 `runtime.Gosched->runtime.gosched_m -> runtime.goschedImpl`
4. 系统监控 `runtime.sysmon -> runtime.retake -> runtime.preemptone`

**主动挂起**

`runtime.gopark` 是触发调度最常见的方法，该函数会将当前 Goroutine 暂停，被暂停的任务不会放回运行队列
```go
func gopark(unlockf func(*g, unsafe.Pointer) bool, lock unsafe.Pointer, reason waitReason, traceEv byte, traceskip int) {
	mp := acquirem()
	gp := mp.curg
	mp.waitlock = lock
	mp.waitunlockf = unlockf
	gp.waitreason = reason
	mp.waittraceev = traceEv
	mp.waittraceskip = traceskip
	releasem(mp)
	mcall(park_m)
}
```
上述会通过 `runtime.mcall` 切换到 `g0` 的栈上调用 `runtime.park_m`：
```go
func park_m(gp *g) {
	_g_ := getg()

	casgstatus(gp, _Grunning, _Gwaiting)
	dropg()

	schedule()
}
```
`runtime.park_m` 会将当前 Goroutine 的状态从 `_Grunning` 切换至 `_Gwaiting`，调用 `runtime.dropg` 移除线程和 Goroutine 之间的关联，在这之后就可以调用 `runtime.schedule` 触发新一轮的调度了。

当 Goroutine 等待的特定条件满足后，运行时会调用 `runtime.goready ` 将因为调用 `runtime.gopark` 而陷入休眠的 Goroutine 唤醒。
```go
func goready(gp *g, traceskip int) {
	systemstack(func() {
		ready(gp, traceskip, true)
	})
}

func ready(gp *g, traceskip int, next bool) {
	_g_ := getg()

	casgstatus(gp, _Gwaiting, _Grunnable)
	runqput(_g_.m.p.ptr(), gp, next)
	if atomic.Load(&sched.npidle) != 0 && atomic.Load(&sched.nmspinning) == 0 {
		wakep()
	}
}
```
`runtime.ready` 会将准备就绪的 Goroutine 的状态切换至 `_Grunnable` 并将其加入处理器的运行队列中，等待调度器的调度。

**系统调用**

系统调用也会触发运行时调度器的调度，为了处理特殊的系统调用，我们甚至在 Goroutine 中加入了 `_Gsyscall` 状态，Go 语言通过 `syscall.Syscall` 和 `syscall.RawSyscall` 等使用汇编语言编写的方法封装操作系统提供的所有系统调用

```go
#define INVOKE_SYSCALL	INT	$0x80

TEXT ·Syscall(SB),NOSPLIT,$0-28
	CALL	runtime·entersyscall(SB)
	...
	INVOKE_SYSCALL
	...
	CALL	runtime·exitsyscall(SB)
	RET
ok:
	...
	CALL	runtime·exitsyscall(SB)
	RET
```
在通过汇编指令 `INVOKE_SYSCALL` 执行系统调用前后，上述函数会调用运行时的 `runtime.entersyscall` 和 `runtime.exitsyscall`，正是这一层包装能够让我们在陷入系统调用前触发运行时的准备和清理工作。

不过出于性能的考虑，如果这次系统调用不需要运行时参与，就会使用 `syscall.RawSyscall` 简化这一过程，不再调用运行时函数。

由于直接进行系统调用会阻塞当前的线程，所以只有可以立刻返回的系统调用才可能会被设置成 `RawSyscall` 类型

**进入系统调用的准备工作**

`runtime.entersyscall` 会在获取当前程序计数器和栈位置之后调用 `runtime.reentersyscall`,
它会完成 Goroutine 进入系统调用前的准备工作：
```go
func reentersyscall(pc, sp uintptr) {
	_g_ := getg()
	_g_.m.locks++
	_g_.stackguard0 = stackPreempt
	_g_.throwsplit = true

	save(pc, sp)
	_g_.syscallsp = sp
	_g_.syscallpc = pc
	casgstatus(_g_, _Grunning, _Gsyscall)

	_g_.m.syscalltick = _g_.m.p.ptr().syscalltick
	_g_.m.mcache = nil
	pp := _g_.m.p.ptr() // 指向p的指针
	pp.m = 0
	_g_.m.oldp.set(pp)
	_g_.m.p = 0
	atomic.Store(&pp.status, _Psyscall)
	if sched.gcwaiting != 0 {
		systemstack(entersyscall_gcwait)
		save(pc, sp)
	}
	_g_.m.locks--
}
```
1. 禁止线程上发生的抢占，防止出现内存不一致的问题；
2. 保证当前函数不会触发栈分裂或者增长；
3. 保存当前的程序计数器 PC 和栈指针 SP 中的内容；
4. 将 Goroutine 的状态更新至 `_Gsyscall`；
5. 将 Goroutine 的处理器(p)和线程(m)暂时分离并更新处理器的状态到 `_Psyscall`；

**恢复工作**
当系统调用结束后,会调用退出系统调用的函数`runtime.exitsyscall`，为当前 Goroutine 重新分配资源，该函数有两个不同的执行路径：

1. 调用 `runtime.exitsyscallfast`；
2. 切换至调度器的 Goroutine 并调用 `runtime.exitsyscall0`；

```go
func exitsyscall() {
	_g_ := getg()

	oldp := _g_.m.oldp.ptr()
	_g_.m.oldp = 0
	if exitsyscallfast(oldp) {
		_g_.m.p.ptr().syscalltick++
		casgstatus(_g_, _Gsyscall, _Grunning)
		...

		return
	}

	mcall(exitsyscall0)
	_g_.m.p.ptr().syscalltick++
	_g_.throwsplit = false
}
```
这两种不同的路径会分别通过不同的方法查找一个用于执行当前 Goroutine 处理器 P，
快速路径(`runtime.exitsyscallfast`)

1. 如果 Goroutine 的原处理器处于 _Psyscall 状态，会直接调用 wirep 将 Goroutine 与处理器进行关联；
2. 如果调度器中存在闲置的处理器，会调用 `runtime.acquirep` 使用闲置的处理器处理当前 Goroutine；

另一个相对较慢的路径 `runtime.exitsyscall0` 会将当前 Goroutine 切换至 `_Grunnable` 状态，并移除线程 M 和当前 Goroutine 的关联：

无论哪种情况，我们在这个函数中都会调用 runtime.schedule 触发调度器的调度

协作式调度

`runtime.Gosched` 函数会主动让出处理器，允许其他 Goroutine 运行。该函数无法挂起 Goroutine，调度器可能会将当前 Goroutine 调度到其他线程上：

```go
func Gosched() {
	checkTimeouts()
	mcall(gosched_m)
}

func gosched_m(gp *g) {
	goschedImpl(gp)
}

func goschedImpl(gp *g) {
	casgstatus(gp, _Grunning, _Grunnable)
	dropg()
	lock(&sched.lock)
	globrunqput(gp)
	unlock(&sched.lock)

	schedule()
}
```

经过连续几次跳转，我们最终在 `g0` 的栈上调用 `runtime.goschedImpl`，运行时会更新 Goroutine 的状态到 `_Grunnable`，让出当前的处理器并将 Goroutine 重新放回全局队列，在最后，该函数会调用 `runtime.schedule` 触发调度。

## 线程管理

Go 语言的运行时会通过调度器改变线程的所有权，它也提供了 `runtime.LockOSThread` 和 `runtime.UnlockOSThread` 让我们有能力绑定 Goroutine 和线程完成一些比较特殊的操作。

`runtime.LockOSThread` 会通过如下所示的代码绑定 Goroutine 和当前线程：
```go
func LockOSThread() {
	if atomic.Load(&newmHandoff.haveTemplateThread) == 0 && GOOS != "plan9" {
		startTemplateThread()
	}
	_g_ := getg()
	_g_.m.lockedExt++
	dolockOSThread()
}

func dolockOSThread() {
	_g_ := getg()
	_g_.m.lockedg.set(_g_)
	_g_.lockedm.set(_g_.m)
}

func UnlockOSThread() {
	_g_ := getg()
	if _g_.m.lockedExt == 0 {
		return
	}
	_g_.m.lockedExt--
	dounlockOSThread()
}

func dounlockOSThread() {
	_g_ := getg()
	if _g_.m.lockedInt != 0 || _g_.m.lockedExt != 0 {
		return
	}
	_g_.m.lockedg = 0
	_g_.lockedm = 0
}
```

## 线程生命周期

Go 语言的运行时会通过 `runtime.startm` 启动线程来执行处理器 P，如果我们在该函数中没能从闲置列表中获取到线程 M 就会调用 `runtime.newm` 创建新的线程：
```go
func newm(fn func(), _p_ *p, id int64) {
	mp := allocm(_p_, fn, id)
	mp.nextp.set(_p_)
	mp.sigmask = initSigmask
	...
	newm1(mp)
}

func newm1(mp *m) {
	if iscgo {
		...
	}
	newosproc(mp)
}
```

创建新的线程需要使用如下所示的 `runtime.newosproc`，该函数在 Linux 平台上会通过系统调用 `clone` 创建新的操作系统线程，它也是创建线程链路上距离操作系统最近的 Go 语言函数：

使用系统调用 `clone` 创建的线程会在线程主动调用 `exit`、或者传入的函数 `runtime.mstart` 返回会主动退出，`runtime.mstart` 会执行调用 `runtime.newm` 时传入的匿名函数 `fn`，到这里也就完成了从线程创建到销毁的整个闭环。