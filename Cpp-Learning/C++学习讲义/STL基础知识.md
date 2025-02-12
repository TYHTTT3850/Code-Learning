# STL基础知识

STL(Standard Template Library) —— 每个字母分别代表标准、模板、库。

STL不是面向对象的编程——而是一种不同的编程模式——通用编程技术。

## STL优点

不需要额外安装。

高可重用性，高移植性，跨平台性。

将数据和操作分离，数据由容器类别加以管理。

程序员不需要直到它的据体实现过程，只需要可以熟练使用即可。

## 六大组件

STL提供了六大组件，彼此之间可以组合套用。

容器、算法、迭代器、仿函数、适配器、空间配置器。

### 容器

各种数据结构，用来存放数据，实现角度来看，STL容器实际上是一种类模板。

### 算法

各种各样的算法，功能模板。

### 迭代器

容器和算法的胶着剂，分为五种类型。

每个容器都有自己的迭代器。

### 仿函数

行为类似函数，可作为算法的某种策略，从实现角度来看，仿函数是一种重载了 `operator()` 的 `class`或者是类模板。

### 适配器

修饰容器或是仿函数或迭代器接口的东西

### 空间配置器

负责空间的配置和管理。

## 三大组件

即前三个STL组件(容器、算法、迭代器) ，是STL广义上的分类，主要重点就集中在这三个上。

 三大组件的关系——容器和算法通过迭代器来进行连接，算法通过迭代器访问容器中的元素。

实际上，所谓容器是存储数据的地方，而算法是操作，操作需要对数据进行，这时候需要迭代器来使这两个联系起来。

### 容器

任何数据结构都是为了实现某种特定算法，STL容器就是将运用最广泛的一些数据结构实现出来。

例如——数组、链表、树、栈、队列、集合、映射表。

#### 分类

序列式：需要注意数据存储时的数据顺序，每个元素均有固定的位置，除非用删除或是插入改变这个位置，`vector` ，`deque` ，`list`等。

关联式：不需要关心其顺序，特点是通过键值对存储元素，并提供快速查找、插入和删除操作，通常按键进行排序或通过哈希表存储。例如 `set` 和 `map` 。

### 迭代器(iterator)

#### 定义

迭代器是一个接口，指向容器的数据，即通过迭代器可以访问(读写)容器中的元素，因此它在某方面很像指针。算法通过这个迭代器去实现对容器的操作，相当于一个容器和算法的粘合剂。

`iterator` 是 STL 中定义的一个类型，通常是指向容器元素的指针或类似指针的对象。

不同的容器通常有自己的迭代器，迭代器的类型是一个通过 `typedef` 定义的类型别名—— `iterator` ，作用域为整个类。

在C++中，`typedef` 用来给现有类型起一个新的名字。通过 `typedef` 可以为某些复杂的类型创建更简洁或有意义的名称。

#### 分类

正向迭代器：`容器类名::iterator  迭代器名`

常量正向迭代器：`容器类名::const_iterator  迭代器名`

反向迭代器：`容器类名::reverse_iterator  迭代器名`

常量反向迭代器：`容器类名：：const_reverse_iterator  迭代器名`

#### 用法

前面已经说过，它指向某个元素，可以看为指针，那 `*迭代器名` 就可以表示迭代器所指的元素了。而每一个容器都有它自己的迭代器，根据不同迭代器指向的位置的不同，可以进行不同的用途。

## 常用容器

### `vector` 容器

#### 介绍：

`vector` 是将元素置于一个动态数组中加以管理的容器，可以添加元素和删除元素。

`vector` 可以随机存取元素。

`vector` 尾部添加或移除元素非常快速。但是在中部或头部插入元素或移除元素比较费时。

#### `vector` 构造/初始化

##### 1、默认构造：

`vector<类型> 容器名` 。

`vector<类型> 容器名{数据}` 。

##### 2、带参数的构造：

`vector<类型> 容器名(n)` ：n代表这个数组长度为n。默认初始化为n个0 。

`vector<类型> 容器名(n,value)`：初始化为n个指定值。

 `vector<类型> 容器名(begin,end)` ：用地址区间 `begin` 到 `end` 里的元素来初始化，区间是左闭右开。

##### 3、拷贝构造

用另一个 `vector` 容器来初始化。

`vector<类型> b(a)` ：要求 `a` 和 `b` 的数据类型必须相同。

`vector<类型> c=a` ：效果和要求同上。

```cpp
#include <vector>

// 仅创建一个list对象
vector<int> v1;

// 初始化为 1,2,3,4,5
vector<int> v2 {1, 2, 3, 4, 5};

// 创建一个有5个元素的list，默认初始化为0
vector<int> v3(5);

// 全部初始化为1
vector<int> v4(5,1);

// 用 begin 到 end 区间里的元素来初始化，其中区间是左闭右开
vector<int> v5(v2.begin(),v2.end())
    
// 用另一个list来初始化，注意类型相同
vector<int> v6(v2);

// 同上
vector<int> v7 = v2;

//二维vector容器
vector<int> a[10]; //定义了一个包含 10 个元素的数组，每个元素都是一个 vector<int> 类型
vector<vector<int>> b; //定义了一个 vector 容器，每个元素都是 vector<int> 类型
vector<vector<int>> c(n,vector<int>(m,0)) //有n个 vector 容器，每个容器都有 m 个元素且初始化为0
```

#### `vector` 迭代器

`容器名.begin()` ：返回首元素的迭代器——地址。

`容器名.end()` ：返回最后一个元素的后一个位置的迭代器——地址。

#### `vector` 常用方法

##### 大小

`容器名.empty()` ：判断是否为空，空则返回真，反之返回假。

`容器名.resize(n)` ：重新指定大小为n。如果重新指定的比原来长，默认用0填充新的位置，如果重新指定的比原来短，超出部分会删除掉。

`容器名.resize(n,value)` ：用指定值填充新的位置。

`容器名.size()` ：返回容器中当前存储的元素个数，即容器中实际包含的元素数量。

##### 赋值

`容器名.assign(begin,end)` ： 将地址区间 `begin` 到 `end` 中的数据拷贝赋值给容器，区间为左闭右开。

`容器名.assign(n,value)` ：将n个value拷贝赋值给容器。

`容器名.swap(another_vector)` ：将另一个 `vector` 容器的值与自身的值互换。

##### 插入和删除

`容器名.push_back(value)` ：在尾部插入指定值。

`容器名.pop_back()` ：删除最后一个值。

`容器名.insert(iterator position,value)` ：在迭代器指向位置插入指定值。

`容器名.insert(iterator position,int count,value)` ：在迭代器指向位置插入 count 个指定值。

`容器名.insert(iterator position,begin,end)` ：指定处插入地址区间 `begin` 到 `end` 的值，左闭右开。

`容器名.erase(iterator position)` ：删除迭代器指向的元素。

`容器名.erase(iterator begin, iterator end)` ：删除迭代器从begin到end之间的元素。

`容器名.clear()` ：是清空所有元素。

##### 访问

`容器名.front()` ：返回容器中第一个值。

`容器名.back()` ：返回容器中最后一个值。

`容器名[index]` ：返回对应索引处的值。

`容器名.at(int index)` ：返回索引 `index` 所指的数据。等同于使用运算符 `[]` 索引。

```cpp
vector<int> v = {1, 2, 3, 4, 5};

// 删除第二个元素 (值为 2)
v.erase(v.begin() + 1);

// 删除从第二个元素到第四个元素
v.erase(v.begin() + 1, v.begin() + 4); // 左闭右开区间
```

#### `vector` 遍历

普通的 `for` 循环

```cpp
vector<int> v = {1, 2, 3, 4, 5};

for (size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    } // size_t 为无符号整数
```

使用范围基的 `for` 循环(C++11 及以上)

```cpp
vector<int> v = {1, 2, 3, 4, 5};

for (int num : v) {
        cout << num << " ";
    }

// 引用的方式
for (int& num : v) {
        cout << num <<" ";  // 修改元素的值
    }
```

通过迭代器实现元素的遍历

```cpp
vector<int> v = {1, 2, 3, 4, 5};

for (vector<int>::iterator it = v.begin(); it != v.end(); it++)
	{
		cout << *it << endl;
	}

vector<int>::iterator it = v.begin();
while (it != v.end())
{
	cout << *it << " ";
	it++;
}
```

### `list` 容器

#### 介绍：

`list` 容器将数据链式存储起来，可以对任意位素进行快速插入或删除元素。

`list` 容器遍历速度，没有数组快。

`list` 占用空间比数组大。

#### `list` 构造/初始化：

##### 1、默认构造：

`list<类型> 容器名` 。

`list<类型> 容器名{数据}` 。

##### 2、带参数的构造：

`list<类型> 容器名(n)` ，n代表这个链表长度为n。默认初始化为0。

`vector<类型> 容器名(n,value)`，初始化为n个指定值。

 `list<类型> 容器名(begin,end)` ，用地址区间 `begin` 到 `end` 里的元素来初始化，区间是左闭右开。

##### 3、拷贝构造

用另一个 `list` 容器来初始化。

`list<类型> b(a)` ：要求 `a` 和 `b` 的数据类型必须相同。

`vector<类型> c=a` ：效果和要求同上。

```cpp
#include <list>

// 仅创建一个list对象
list<int> l1;

// 初始化为 1,2,3,4,5
list<int> l2 {1,2,3,4,5};

// 创建一个有5个元素的list，默认初始化为0
list<int> l3(5);

// 全部初始化为1
list<int> l4(5,1);

// 用 begin 到 end 区间里的元素来初始化，其中区间是左闭右开
list<int> l5(l2.begin(),l2.end())

// 用另一个list来初始化，注意类型相同
list<int> l6(l2);

// 同上
list<int> l7 = l2;
```

#### `list` 迭代器

`容器名.begin()` ：返回首元素的迭代器——地址。

`容器名.end()` ：返回最后一个元素的后一个位置的迭代器——地址。

#### `list` 常用方法

##### 大小

`容器名.empty()` ：判断是否为空，空则返回真，反之返回假。

`容器名.resize(n)` ：重新指定大小为n。如果重新指定的比原来长，默认用0填充新的位置；如果重新指定的比原来短，超出部分会删除掉。

`容器名.resize(n,value)` ：用指定值填充新的位置。

`容器名.size()` ：返回容器中当前存储的元素个数，即容器中实际包含的元素数量。

##### 赋值

`assign(begin,end)` ：将地址区间 `begin` 到 `end` 中的数据拷贝赋值给容器。

`assign(n,value)` ：将n个value拷贝赋值给容器。

`swap(another_list)` 将另一个 `list` 容器与自身的值互换。

##### 插入和删除

`容器名.push_back(value)` ：在尾部插入指定值。

`容器名.pop_back()` ：删除最后一个值。

`容器名.push_front(value)` ：在容器开头插入一个元素。

`容器名.pop_front()` ：删除第一个元素。

`容器名.insert(iterator position,value)` ：在迭代器指向位置插入指定值。

`容器名.insert(iterator position,int count,value)` ：在迭代器指向位置插入 count 个指定值。

`容器名.insert(iterator position,begin,end)` ：指定处插入地址区间 `begin` 到 `end` 的值，左闭右开。

`容器名.erase(iterator position)` ：删除迭代器指向的元素。

`容器名.erase(iterator begin, iterator end)` ：删除迭代器从begin到end之间的元素。

`容器名.remove(value)` ：删除容器中所有与指定值匹配的元素。

`容器名.clear()` ：是清空所有元素。

##### 访问

`容器名.front()` ：返回容器中第一个元素。

`容器名.back()` ：返回容器中最后一个元素。

无法向 `vector` 容器一样直接通过下标访问。

##### 排序

`容器名.reverse()` ：反转链表。

`容器名.sort()` ：链表排序。

#### `list` 遍历

只能使用范围基和迭代器遍历。和 `vector` 容器基本一样。

### `deque` 容器

### `stack` 容器

### `queue` 容器

### `string` 容器

### `set/multiset` 容器

### `map/multimap` 容器