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

容器可分类为关联式和序列式

|                        类型                        |                             容器                             |
| :------------------------------------------------: | :----------------------------------------------------------: |
| 序列式(按照**元素的插入顺序存储数据**，不自动排序) |  `vector` 、`list` 、`string` 、`queue` 、`deque` 、`stack`  |
|        关联式(会**自动排序或进行哈希存储**)        | `set\multiset` 、`map\multimap` 、`unordered_set\unordered_multiset` 、`unordered_map\unordered_multimap` |



### `vector` 容器

#### 介绍

`vector` 是将元素置于一个动态数组中加以管理的容器，可以添加元素和删除元素。

`vector` 可以随机存取元素。

`vector` 尾部添加或移除元素非常快速。但是在中部或头部插入元素或移除元素比较费时。

<a id="`vector`-构造/初始化"></a>

#### `vector` 构造/初始化

##### 普通构造

`vector<类型> 容器名` 。

`vector<类型> 容器名{数据}` 。

`vector<类型> 容器名 = {数据}` 。

##### 带参数的构造

`vector<类型> 容器名(count)` ：初始化大小为count，用默认值填充。

`vector<类型> 容器名(count,value)`：初始化为count个指定值。

 `vector<类型> 容器名(begin,end)` ：用地址区间 `begin` 到 `end` 里的元素来初始化，区间是左闭右开。

##### 拷贝构造

用另一个 `vector` 容器来初始化。

`vector<类型> b(a)` ：要求 `a` 和 `b` 的数据类型必须相同。

```cpp
#include <vector>

// 仅创建一个list对象
vector<int> v1;

// 初始化为 1,2,3,4,5
vector<int> v2 {1, 2, 3, 4, 5};
vector<int> v2_2 = {1,2,3,4,5};

// 创建一个有5个元素的list，默认初始化为0
vector<int> v3(5);

// 全部初始化为1
vector<int> v4(5,1);

// 用 begin 到 end 区间里的元素来初始化，其中区间是左闭右开
vector<int> v5(v2.begin(),v2.end())
    
// 用另一个vector来初始化，注意类型相同
vector<int> v6(v2);

同上
vector<int> v7 = v2;

//二维vector容器
vector<int> a[10]; //定义了一个包含 10 个元素的数组，每个元素都是一个 vector<int> 类型
vector<vector<int>> b; //定义了一个 vector 容器，每个元素都是 vector<int> 类型
vector<vector<int>> c(5,vector<int>(3,0)) //有5个 vector 容器，每个容器都有3个元素且初始化为0
```

<a id="`vector`-迭代器"></a>
#### `vector` 迭代器

##### 普通迭代器

**从左往右(正向迭代)**：

`容器名.begin()` ：返回首元素(**最左端**)的迭代器——地址。

`容器名.end()` ：返回最后一个元素(**最右端**)的后一个位置的迭代器——地址。

**从右往左(反向迭代)**：

`容器名.rbegin()` ：返回首元素(**最右端**)的迭代器——地址。

`容器名.rend()` ：返回最后一个元素(**最左端**)的后一个位置的迭代器——地址。

##### 常量迭代器

不能修改元素，其余和普通迭代器一样。

`容器名.cbegin()` 、 `容器名.cend()` 、 `容器名.rbegin()` 、 `容器名.rend()` 。

#### `vector` 常用方法

<a id="`vector`-大小"></a>
##### 大小

`容器名.empty()` ：判断是否为空，空则返回真，反之返回假。

`容器名.size()` ：返回容器中当前存储的元素个数，即容器中实际包含的元素数量。

`容器名.resize(count)` ：重新指定大小为count。如果重新指定的比原来长，用默认值填充新的位置，如果重新指定的比原来短，超出部分会删除掉。

`容器名.resize(count,value)` ：用指定值填充新的位置。

<a id="`vector`-赋值"></a>
##### 赋值

`容器名.assign(begin,end)` ： 将地址区间 `begin` 到 `end` 中的数据拷贝赋值给容器，区间为左闭右开。

`容器名.assign(count,value)` ：将count个指定值拷贝赋值给容器。

`容器名.swap(another_vector)` ：将另一个 `vector` 容器的值与自身的值互换。

`vector<类型> c=a` ：重载 `=` 操作符，将另一个 `vector` 容器的数据复制到自身。

##### 插入和删除

`容器名.push_back(value)` ：在尾部插入指定值。

`容器名.pop_back()` ：删除最后一个值。

`容器名.insert(iterator_position,value)` ：在迭代器指向位置插入指定值。

`容器名.insert(iterator_position,int count,value)` ：在迭代器指向位置插入 count 个指定值。

`容器名.insert(iterator_position,begin,end)` ：指定处插入地址区间 `begin` 到 `end` 的值，左闭右开。

`容器名.erase(iterator_position)` ：删除迭代器指向的元素。

`容器名.erase(iterator_begin, iterator_end)` ：删除迭代器从begin到end之间的元素。

`容器名.clear()` ：清空所有元素。

```cpp
vector<int> v = {1, 2, 3, 4, 5};

// 删除第二个元素 (值为 2)
v.erase(v.begin() + 1);

// 删除从第二个元素到第四个元素
v.erase(v.begin() + 1, v.begin() + 4); // 左闭右开区间
```

<a id="`vector`-索引"></a>
##### 索引

`容器名.front()` ：返回容器中第一个值。

`容器名.back()` ：返回容器中最后一个值。

`容器名[index]` ：返回对应索引处的值。

`容器名.at(int index)` ：返回索引 `index` 所指的数据。等同于使用运算符 `[]` 索引。

<a id="`vector`-遍历"></a>
#### `vector` 遍历

普通的 `for` 循环

```cpp
vector<int> v = {1, 2, 3, 4, 5};

for (size_t i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
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

#### 介绍

`list` 容器将数据链式存储起来，可以对任意位素进行快速插入或删除元素。

`list` 容器遍历速度，没有数组快。

`list` 占用空间比数组大。

<a id="`list`-构造/初始化"></a>
#### `list` 构造/初始化

##### 普通构造

`list<类型> 容器名` 。

`list<类型> 容器名{数据}` 。

`list<类型> 容器名 = {数据}` 。

##### 带参数的构造

`list<类型> 容器名(count)` ：初始化大小为count，用默认值填充。

`list<类型> 容器名(count,value)` ：初始化为count个指定值。

 `list<类型> 容器名(begin,end)` ：用地址区间 `begin` 到 `end` 里的元素来初始化，区间是左闭右开。

##### 拷贝构造

用另一个 `list` 容器来初始化。

`list<类型> b(a)` ：要求 `a` 和 `b` 的数据类型必须相同。

```cpp
#include <list>

// 仅创建一个list对象
list<int> l1;

// 初始化为 1,2,3,4,5
list<int> l2 {1,2,3,4,5};
list<int> l2_2 = {1,2,3,4,5};

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

<a id="`list`-迭代器"></a>

#### `list` 迭代器

##### 普通迭代器

**从左往右(正向迭代)**：

`容器名.begin()` ：返回首元素(**最左端**)的迭代器——地址。

`容器名.end()` ：返回最后一个元素(**最右端**)的后一个位置的迭代器——地址。

**从右往左(反向迭代)**：

`容器名.rbegin()` ：返回首元素(**最右端**)的迭代器——地址。

`容器名.rend()` ：返回最后一个元素(**最左端**)的后一个位置的迭代器——地址。

##### 常量迭代器

不能修改元素，其余和普通迭代器一样。

`容器名.cbegin()` 、 `容器名.cend()` 、 `容器名.rbegin()` 、 `容器名.rend()` 。

#### `list` 常用方法

<a id="`list`-大小"></a>
##### 大小

`容器名.empty()` ：判断是否为空，空则返回真，反之返回假。

`容器名.size()` ：返回容器中当前存储的元素个数，即容器中实际包含的元素数量。

`容器名.resize(count)` ：重新指定大小为count。如果重新指定的比原来长，用默认值填充新的位置；如果重新指定的比原来短，超出部分会删除掉。

`容器名.resize(count,value)` ：用指定值填充新的位置。

<a id="`list`-赋值"></a>
##### 赋值

`容器名.assign(begin,end)` ：将地址区间 `begin` 到 `end` 中的数据拷贝赋值给容器。

`容器名.assign(count,value)` ：将count个value拷贝赋值给容器。

`容器名.swap(another_list)` 将另一个 `list` 容器与自身的值互换。

`list<类型> c=a` ：重载 `=` 操作符，将另一个 `list` 容器的数据复制到自身。

<a id="`list`-插入和删除"></a>
##### 插入和删除

`容器名.push_back(value)` ：在尾部插入指定值。

`容器名.pop_back()` ：删除最后一个值。

`容器名.push_front(value)` ：在容器开头插入一个元素。

`容器名.pop_front()` ：删除第一个元素。

`容器名.insert(iterator_position,value)` ：在迭代器指向位置插入指定值。

`容器名.insert(iterator_position,int count,value)` ：在迭代器指向位置插入 count 个指定值。

`容器名.insert(iterator_position,begin,end)` ：指定处插入地址区间 `begin` 到 `end` 的值，左闭右开。

`容器名.erase(iterator_position)` ：删除迭代器指向的元素。

`容器名.erase(iterator_begin, iterator_end)` ：删除迭代器从begin到end之间的元素。

`容器名.remove(value)` ：删除容器中所有与指定值匹配的元素。

`容器名.clear()` ：清空所有元素。

##### 索引

`容器名.front()` ：返回容器中第一个元素。

`容器名.back()` ：返回容器中最后一个元素。

无法向 `vector` 容器一样直接通过下标访问。

##### 排序

`容器名.reverse()` ：反转链表。

`容器名.sort()` ：链表排序。

#### `list` 遍历

只能使用范围基和迭代器遍历，与[ `vector` 遍历](#`vector`-遍历)基本一致。

### `string` 容器

#### 介绍

字符：单个字，如 `a` ，`我` 。在C++中他们为 `char` 型。

字符串：多个字符组成的串。

`string` 容器和C语言中的 `char*` 很像，而 `string` 是C++风格的字符串，`string` 本质上是一个类。

`string` 内部封装了 `char*` ，并提供了更丰富和安全的操作。

#### `string` 构造/初始化

##### 普通构造

`string 容器名` 。

`string 容器名 = "字符串"` 。

##### 带参数的构造

`string 容器名(count,value)`：初始化为count个指定值，指定值只能为单个字符。

##### 拷贝构造

用另一个 `string` 容器来初始化。

`string b(a)` ：也可以直接 `string b("字符串")`

```cpp
#include <string>

// 仅创建一个string对象
string s1;

// 初始化为"hello"
string s2 = "hello";

// 初始化为5个'a'
string s3(5,'a');

// 用另一个string来初始化
string s4(s2);

// 同上
string s5 = s2;
```

#### `string` 迭代器

与 [`vector` 迭代器](#`vector`-迭代器)和 [`list` 迭代器](#`list`-迭代器)基本一致。

#### `string` 常用方法

##### 大小

`容器名.empty()` ：判断是否为空，空则返回真，反之返回假。

`容器名.size()` ：返回字符中字符的个数(不包括 `\0` 终止符)。

`容器名.resize(count)` ：重新指定大小为count。如果重新指定的比原来长，用默认值填充新的位置；如果重新指定的比原来短，超出部分会删除掉。

`容器名.resize(count,value)` ：用指定字符填充新的位置。

##### 赋值

`容器名.assign(count,value)` ：将count个字符赋值给此字符串。

`容器名.assign(another_string)` ：将另一个字符串赋值给此字符串。

`容器名.assign(another_string,n)` ：将另一个字符串从**索引n**处到末尾的字符赋值给此字符串。

`容器名.assign(another_string,n,count)` ：将另一个字符串从**索引n**处取count个字符赋值给此字符串。

`容器名.assign(字符串字面量,count)` ：取字符串字面量的前count个字符赋值给此字符串。

`容器名.swap(another_string)` 将另一个 `string` 容器与自身的值互换。

`string c=a` ：重载 `=` 操作符，将另一个 `string` 容器的数据复制到自身。

##### 拼接

运算符 `+` ：连接两个字符串并返回一个新的字符串，不会修改原来的字符串。

运算符 `+=` ：将右侧的字符串附加到左侧的字符串上。修改原有字符串，不会创建新的对象。

`容器名.append(another_string)` ：将另一个字符串拼接到此字符串。

`容器名.append(another_string,n)` ：将另一个字符串从**索引n**处到末尾的字符拼接到此字符串。

`容器名.append(another_string,n,count)` ：将另一个字符串从**索引n**处取count个字符拼接到此字符串。

`容器名.append(字符串字面量,count)` ：取字符串字面量的前count个字符拼接到此字符串。默认整个拼接。

##### 查找和替换

**find从左往右查找第一次出现的位置，rfind从右往左查找第一次出现的位置**，`-1` 表示未找到。

`容器名.find(another_string,n)` ：返回从**索引n**处开始查找指定字符串第一次出现位置。

`容器名.find(char,n)` ：返回从**索引n**处开始查找指定字符第一次出现位置。

`容器名.find(字符串字面量,n)` ：返回从**索引n**处开始查找指定字符串字面量第一次出现位置。

`容器名.find(字符串字面量,n,count)` ：返回从**索引n**处开始查找指定字符串字面量的前count个字符第一次出现位置。

`容器名.rfind()` ：传入参数同 `find()` ，返回的位置相当于最后一次出现的位置(右往左第一次出现)。

`容器名.replace(n,count,another_string)` ：替换从**索引n**处开始count个字符为指定字符串。

`容器名.replace(n,count,字符串字面量)` ：替换从**索引n**处开始count个字符为指定字符串字面量。

##### 比较

字符串比较是按字符的ASCII码进行对比：`=` 则返回0，`>` 则返回1，`<` 则返回-1。

`容器名.compare(another_string)` ：与另一个字符串比较。

`容器名.compare(字符串字面量)` ：与另一个字符串字面量比较。

##### 索引

`容器名.front()` ：返回字符串中第一个字符。

`容器名.back()` ：返回字符串中最后一个字符(不是 `\0` )。

`容器名[index]` ：返回对应索引处的字符。

`容器名.at(int index)` ：返回索引 `index` 所指的字符。等同于使用运算符 `[]` 索引。

##### 插入和删除

`容器名.insert(n,字符串字面量)` ：在**索引n**处插入指定的字符串字面量。

`容器名.insert(n,another_string)` ：在**索引n**处插入指定的字符串。

`容器名.insert(n,count,char)` ：在**索引n**处插入count个指定的字符。

`容器名.erase(n)` ：删除从**索引n处**开始到结尾的字符。

`容器名.erase(n,count)` ：删除从**索引n处**开始的count个字符。

`容器名.clear()` ：清空字符串

##### 子串

`容器名.substr(n,count)` ：返回由开始**索引n**处开始的count个字符组成的字符串。

#### `string` 遍历

与[ `vector` 遍历](#`vector`-遍历)基本一致。

### `queue` 容器

`queue` 是一种先进先出的数据结构，俗称队列。

队列容器允许从**队尾端新增元素**，从**队头端移除元素**。

队列中只有**队头和队尾才可以被外界使用**，因此队列**不允许有遍历行为**。

#### `queue` 构造/初始化

##### 普通构造

`queue<类型> 容器名` 。

##### 拷贝构造

用另一个 `queue` 容器来初始化。

`queue<类型> b(a)` ：要求 `a` 和 `b` 的数据类型必须相同。

##### 使用其他容器来初始化

可以使用 `list` ，`deque` ，来实现初始化，注意数据类型要匹配。

`queue<类型> 容器名(deque型容器)` 。

`queue<类型,list<类型>> 容器名(list型容器)` 。

注意 `deque` 和 `list` 的区别。

```cpp
#include <list>
#include <deque>
#include <queue>

list<int> l {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
deque<int> d {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// 普通构造
queue<int> q;

// 使用deque容器来初始化
queue<int> q1(d);

// 使用list容器来初始化
queue<int,list<int>> q2(l);

// 拷贝构造
queue<int> q3(q1);

queue<int> q4 = q1;
```

#### `queue` 迭代器

`queue` 本质上是**容器适配器**，没有自己的迭代器。

#### `queue` 常用方法

##### 大小

`容器名.empty()` ：判断队列是否为空。

`容器名.size()` ： 返回队列的元素个数。

##### 赋值

`容器名.swap(another_queue)` 将另一个 `queue` 容器与自身的值互换。

`queue<类型> c=a` ：重载 `=` 操作符，将另一个 `queue` 容器的数据复制到自身。

##### 索引

`容器名.back()` ：返回最后一个元素。

`容器名.front()` ：返回第一个元素。

##### 插入和删除

`容器名.push(value)` ：往队尾添加指定值。

`容器名.pop()` ：从队头移除第一个元素。

### `deque` 容器

#### 介绍

双端队列，即两端都可以添加和移除元素。

结合了 `vector` 容器和 `list` 容器的特点，还有一些额外的特性：

**来自 `vector` 的特点**：

- **支持随机访问**( O(1) 复杂度，`operator[]` 和 `at()` )。
- **连续存储部分数据**(但整体上是分段存储，不保证完全连续)。
- **支持 `push_back()` 和 `pop_back()`，尾部操作高效**( O(1) )。

**来自 `list` 的特点**：

- **支持 `push_front()` 和 `pop_front()` ，头部操作高效**( O(1) )。
- **扩展时不会整体移动所有元素**(不像 `vector` 需要重新分配和拷贝)。

**额外的特性**(区别于 `vector` 和 `list` )：

- **分块存储结构**(不像 `vector` 是单块连续内存，`deque` 由多个小块组成)。
- **比 `list` 更节省空间**( `list` 需要额外存储指针，而 `deque` 仅在块间管理索引)。

#### `deque` 构造/初始化

与[ `vector` 构造/初始化](#`vector`-构造/初始化)和[ `list` 构造/初始化](#`list`-构造/初始化)基本一致。

```cpp
#include <deque>

// 默认构造
deque<int> d;

// 初始化
deque<int> d1 {1, 2, 3, 4, 5};

// 设定大小为5，默认值填充
deque<int> d2(5);

// 大小为5，用1填充
deque<int> d3(5,1);

// 用 begin 到 end 区间里的元素来初始化，其中区间是左闭右开
deque<int> d4(d3.begin(), d3.end());

// 拷贝构造
deque<int> d5(d4);
```

#### `deque` 迭代器

与 [`vector` 迭代器](#`vector`-迭代器)和 [`list` 迭代器](#`list`-迭代器)基本一致。

#### `deque` 常用方法

##### 大小

与[ `vector` 大小](#`vector`-大小)和[ `list` 大小](#`list`-大小)基本一致。

##### 赋值

与[ `vector` 赋值](#`vector`-赋值)和[ `list` 赋值](#`list`-赋值)基本一致。

##### 插入和删除

除 `容器名.remove(value)` 方法外，`deque` 容器没有这种方法，其余和[ `list` 插入和删除](#`list`-插入和删除)基本一致。

##### 索引

与[ `vector` 索引](#`vector`-索引)基本一致。

#### `deque` 遍历

与[ `vector` 遍历](#`vector`-遍历)基本一致。

### `stack` 容器

#### 介绍

`stack` 是一种先进后出的数据结构，只有一个出口，即俗称的栈。

栈**不允许有遍历行为**，只有顶部元素可以使用。

#### `stack` 构造/初始化

##### 普通构造

`stack<类型> 容器名` 。

##### 拷贝构造

用另一个 `stack` 容器来初始化。

`stack<类型> b(a)` ：要求 `a` 和 `b` 的数据类型必须相同。

##### 使用其他容器来初始化

可以使用 `vector` ，`list` ，`deque` ，来实现初始化，注意数据类型要匹配。

`stack<类型> 容器名(deque型容器)` 。

`stack<类型,vector<类型>> 容器名(vector型容器)` 。

`stack<类型,list<类型>> 容器名(list型容器)` 。

注意 `deque` 和 `vector` 、`list` 的区别。

```cpp
#include <stack>
#include <vector>
#include <list>
#include <deque>

vector<int> v {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
list<int> l {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
deque<int> d {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// 普通构造
stack<int> s;

// 使用vector容器来初始化
stack<int,vector<int>> s1(v); //后面的元素在前面的元素的上层，下同

// 使用list容器来初始化
stack<int,list<int>> s2(l);

// 使用deque容器来初始化
stack<int> s3(d);

// 拷贝构造
stack<int> s4(s3);

stack<int> s5 = s1;
```

#### `stack` 迭代器

`stack` 本质上是**容器适配器**，没有自己的迭代器。

#### `stack` 常用方法

##### 大小

`容器名.empty()` ：判断栈是否为空。

`容器名.size()` ： 返回栈的元素个数。

##### 赋值

`容器名.swap(another_stack)` 将另一个 `stack` 容器与自身的值互换。

`stack<类型> c=a` ：重载 `=` 操作符，将另一个 `stack` 容器的数据复制到自身。

##### 索引

`容器名.top()` ：返回栈顶元素。

##### 插入和删除

`容器名.pop()` ：删除栈顶元素。

`容器名.push(value)` ：向栈顶压入指定值。

### `set/multiset` 容器

#### 介绍

所有元素都会在插入时自动被排序，`set` 和 `multset` 都属于关联式容器。

**set和multiset的区别**：`set` 不许容器中有重复的元素。`multiset` 允许容器中有重复的元素。

因为 `set` 插入数据的同时会返回插入结果，表示插入是否成功，`multiset` 不会检测数据，因此可以插入重复数据。

#### `set/multiset` 构造/初始化

##### 普通构造

`set<类型> 容器名` 。

`set<类型> 容器名{数据}` 。

`set<类型> 容器名 = {数据}` 。

`multiset<类型> 容器名` 。

`multiset<类型> 容器名{数据}` 。

`multiset<类型> 容器名 = {数据}` 。

##### 带参数的构造

 `set<类型> 容器名(begin,end)` ：用地址区间 `begin` 到 `end` 里的元素来初始化，区间是左闭右开。

 `multiset<类型> 容器名(begin,end)` ：用地址区间 `begin` 到 `end` 里的元素来初始化，区间是左闭右开。

##### 拷贝构造

**用另一个 `set\multiset` 容器来初始化。**

`set<类型> b(a)` ：要求 `a` 和 `b` 的数据类型必须相同。

`multiset<类型> b(a)` ：要求 `a` 和 `b` 的数据类型必须相同。

```cpp
#include <vector>
#include <set>

vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// 默认构造
set<int> s;
set<int> ms;

// 初始化数据
set<int> s1 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
multiset<int> ms1 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// 用 begin 到 end 区间里的元素来初始化，其中区间是左闭右开
set<int> s2(v.begin(), v.end());
multiset<int> ms2(v.begin(), v.end());

// 拷贝构造
set<int> s3(s1);
multiset<int> ms3(ms1);
```

#### `set/multiset` 迭代器

**注意**：`set/multiset` 迭代器**无论是普通迭代器还是常量迭代器**，**都无法对元素进行修改**。

其余与[ `vector` 迭代器](#`vector`-迭代器)和 [`list` 迭代器](#`list`-迭代器)基本一致。

#### `set/multiset` 常用方法

对 `set/multiset` 都适用。

##### 大小

`容器名.empty()` ：判断是否为空，空则返回真，反之返回假。

`容器名.size()` ：返回容器中当前存储的元素个数，即容器中实际包含的元素数量。

##### 赋值

`容器名.swap(another_set)` ：将另一个 `set` 容器的值与自身的值互换。

`容器名.swap(another_multiset)` ：将另一个 `multiset` 容器的值与自身的值互换。

`set<类型> c=a` ：重载 `=` 操作符，将另一个 `set` 容器的数据复制到自身。

`multiset<类型> c=a` ：重载 `=` 操作符，将另一个 `multiset` 容器的数据复制到自身。

##### 插入和删除

`容器名.insert(value)` ：插入指定值。

`容器名.insert(iterator_position,value)` ：在迭代器指向位置插入指定值。

**注意：即使提供迭代器位置，也只是作为提示，最终仍会按排序规则存储。** 

`容器名.erase(iterator_position)` ：删除迭代器指向的元素。

`容器名.erase(iterator_begin, iterator_end)` ：删除迭代器从begin到end之间的元素。

`容器名.clear()` ：清空所有元素。

##### 索引

不支持索引。

##### 查找与统计

`容器名.find(value)` ：查找指定值是否存在。若存在，返回该元素的迭代器；若不存在，则返回end()迭代器。

`容器名.count(value)` ：统计指定值的数量。

### `map/multimap` 容器

#### 介绍

`map/multimap` 都属于**关联式容器**。

**`pair` 对组**：成对出现的数据，利用对组可以返回两个数据。

`pair` 对组有两种创建方式：

`pair<类型,类型> p(value1,value2)` 。

`pair<类型,类型> p = make_pair(value1,value2)` 。

`pair` 对组的数据：

`对组名.first` ：第一个数据。

`对组名.second` ：第二个数据。

在 `map\multimap` 中，所有元素都是 `pair` 对组。`pair` 对组的第一个元素为**key(键值)**,起到**索引作用**，第二个元素为**value(实值)**，所有元素都会根据**元素的键值**自动排序。

**map和multimap区别：**

map不允许容器中有重复key值元素。

multimap允许容器中有重复key值元素。

#### `map/multimap` 构造/初始化

##### 普通构造

`map<类型,类型> 容器名` 。

`multimap<类型,类型> 容器名` 。

##### 拷贝构造

**用另一个 `map\multimap` 容器来初始化。**

`map<类型,类型> b(a)` ：要求 `a` 和 `b` 的数据类型必须相同。

`multimap<类型,类型> b(a)` ：要求 `a` 和 `b` 的数据类型必须相同。

```cpp
#include <map>

map<string, int> m;
multimap<string, int> mm;
map<string, int> m1(m);
multimap<string, int> mm1(mm);
```

#### `map/multimap` 迭代器

与 [`vector` 迭代器](#`vector`-迭代器)和 [`list` 迭代器](#`list`-迭代器)一样也有**正反向迭代器**和**常量正反向迭代器**。只不过在使用迭代器访问数据时，要指明 `first` 或 `second` 。

#### `map/multimap` 常用方法

对 `set/multiset` 都适用。

##### 大小

`容器名.empty()` ：判断是否为空，空则返回真，反之返回假。

`容器名.size()` ：返回容器中当前存储的元素个数，即容器中实际包含的元素数量。

##### 赋值

`容器名.swap(another_map)` ：将另一个 `map` 容器的值与自身的值互换。

`容器名.swap(another_multimap)` ：将另一个 `multimap` 容器的值与自身的值互换。

`map<类型,类型> c=a` ：重载 `=` 操作符，将另一个 `map` 容器的数据复制到自身。

`multimap<类型,类型> c=a` ：重载 `=` 操作符，将另一个 `multimap` 容器的数据复制到自身。

##### 插入和删除

**注意**：插入和删除的值都为 `pair` 对组。

`容器名.insert(pair_value)` ：插入指定值。

`容器名.insert(iterator_position,pqir_value)` ：在迭代器指向位置插入指定值。

**注意：即使提供迭代器位置，也只是作为提示，最终仍会键值自动排序存储。**

`容器名.erase(iterator_position)` ：删除迭代器指向的元素。

`容器名.erase(iterator_begin, iterator_end)` ：删除迭代器从begin到end之间的元素。

`容器名.erase(key)` ：删除键值为key的 `pair` 对组。

`容器名.clear()` ：清空所有元素。

##### 索引

`map` 支持 `容器名[key]` 和 `容器名.at(key)` 的索引形式。

`multimap` 不支持索引。

##### 查找与统计

`容器名.find(key)` ：查找指定键值是否存在。若存在，返回该键值的元素的迭代器；若不存在，返回end()迭代器。

`容器名.count(key)` ：统计指定键值的数量。

## 函数对象

### 概念

一个类重载了函数调用符号 `()` 使得其类对象可以进行类似于函数调用的操作，因此被称为**函数对象**。

函数对象进行调用操作由于比较像函数调用，所以又被称为**仿函数**。

如果一个类既定义了含初始化列表的构造函数，又重载了 `operator()` ，其解析优先级如下：

1、**第一优先级：解释为变量(对象)定义。**

2、**第二优先级(如果第一条不适用)：考虑仿函数调用。**

例如：

```cpp
type name(parameters); //优先解析为变量(对象)定义

name(parameters); //对象已经确定，解析为仿函数调用
```

**注意**：这里所有的函数对象、仿函数等概念都是指一个**类**！！！

### 区别和共性

由于函数对象并不是真正的函数调用，因此会有几点区别但也有一样的地方：

1.函数对象可以有自己的属性，而普通函数没有。

2.函数对象可以像普通函数调用一样有参数，有返回值。

3.函数对象可以作为参数传递。

<a id="谓词"></a>

## 谓词

谓词就是返回值类型为 `bool` 类型的函数或函数对象。

谓词的实现通常使用一下三种方式：

1、**普通函数**。

2、[**Lambda 表达式**](Lambda匿名函数表达式.md)(常用)。

3、**函数对象(仿函数)**，即通过重载 `operator()` 实现。

传入参数为1个的 —— **一元谓词**。

传入参数为2个的 —— **二元谓词**。

## 标准仿函数

STL 提供了一些内置的函数对象，需要包含头文件：`#include <functional>` 。

**注意**：这里所有的函数对象、仿函数等概念都是指一个**类**！！！

### 算数仿函数

| 仿函数             | 作用         |
| ------------------ | ------------ |
| `plus<类型>`       | 加法 ( `+` ) |
| `minus<类型>`      | 减法 ( `-` ) |
| `multiplies<类型>` | 乘法 ( `*` ) |
| `divides<类型>`    | 除法 ( `/` ) |
| `modulus<类型>`    | 取模 ( `%` ) |
| `negate<类型>`     | 取负 ( `-` ) |

**示例**：

```cpp
#include <iostream>
#include <functional>
using namespace std;

int main() {
    plus<int> add;  // 加法仿函数
    cout << "3 + 5 = " << add(3, 5) << endl; // 输出 8

    negate<int> neg;  // 取负仿函数
    cout << "negate(10) = " << neg(10) << endl; // 输出 -10

    return 0;
}
```

### 关系仿函数

| 仿函数                | 作用             |
| --------------------- | ---------------- |
| `equal_to<类型>`      | 判断相等( `==` ) |
| `not_equal_to<类型>`  | 判断不等( `!=` ) |
| `greater<类型>`       | 大于( `>` )      |
| `greater_equal<类型>` | 大于等于( `>=` ) |
| `less<类型>`          | 小于( `<` )      |
| `less_equal<类型>`    | 小于等于( `<=` ) |

**示例：**

```cpp
#include <iostream>
#include <functional>
using namespace std;

int main() {
    greater<int> greater_than;  // 大于仿函数
    cout << "5 > 3 ? " << greater_than(5, 3) << endl;  // 输出 1 (true)

    equal_to<int> equals;  // 相等仿函数
    cout << "5 == 5 ? " << equals(5, 5) << endl;  // 输出 1 (true)

    return 0;
}
```

### 逻辑仿函数

| 仿函数              | 作用           |
| ------------------- | -------------- |
| `logical_and<类型>` | 逻辑与( `&&` ) |
| `logical_or<类型>`  | 逻辑或( `||` ) |
| `logical_not<类型>` | 逻辑非( `!` )  |

**示例：**

```cpp
#include <iostream>
#include <functional>
using namespace std;

int main() {
    logical_and<bool> land;
    logical_or<bool> lor;
    logical_not<bool> lnot;

    cout << "true && false = " << land(true, false) << endl;  // 输出 0
    cout << "true || false = " << lor(true, false) << endl;  // 输出 1
    cout << "!true = " << lnot(true) << endl;  // 输出 0

    return 0;
}
```

## 常用STL算法

使用STL内置算法一般需要包含头文件是 `#include <algorithm>` 。

### 遍历算法

#### `for_each()` 

对范围内的每个元素执行指定操作。

`for_each(iterator_begin,iterator_end,function)`

三个参数分别是对应：起始的迭代器，终止迭代器，函数或函数对象(指定操作)。

```cpp
//传入lambda函数，乘二输出
for_each(vec.begin(), vec.end(), [](int x) {cout << x * 2 << " ";}); 
```

#### `transform()`

将一个容器里的内容**经指定操作后**搬运到另一个容器中。

`transform(iterator_begin1,iterator_end1,iterator_begin2,function)`

参数分别对应：原容器起始迭代器，原容器终止迭代器，新容器起始迭代器，函数或函数对象(指定操作)。

```cpp
// 平方后拷贝至新容器
transform(vec.begin(), vec.end(), result.begin(), [](int x) { return x * x; });
```

### 查找算法

#### `find()`

查找指定值第一次出现的位置(迭代器)，找不到则返回迭代器end()。

`find(iterator_begin,iterator_end,value)`

参数分别对应着起始迭代器、终止迭代器、指定值。例如：

```cpp
// 查找3
auto it = find(vec.begin(), vec.end(), 3);
```

#### `find_if()`

按条件查找元素，返回第一个满足条件的元素所在位置(迭代器)，找不到则返回迭代器end()。

`find_if(iterator_begin,iterator_end,predicate)`

参数分别为：起始迭代器、终止迭代器、[一元谓词](#谓词)。例如：

```cpp
// 查找2的倍数
auto it = find_if(vec.begin(), vec.end(), [](int x) { return x % 2 == 0; });
```

#### `adjacent_find()`

默认查找相邻且相同元素，返回第一个匹配元素的位置(迭代器)，找不到则返回选代器end();

`adjacent_find(iterator_begin,iterator_end)`

参数分别为：起始迭代器、终止迭代器。例如：

```cpp
// 查找相邻且相同元素
auto it = adjacent_find(vec.begin(), vec.end());
```

也可以传入[二元谓词](#谓词)，自定义比较规则，例如：

```cpp
// 相邻元素之差不超过 3
auto it = adjacent_find(vec.begin(),vec.end(),[](int a,int b) {return abs(a - b) <= 3; });
```

#### `binary_search()`

用于在**已排序的容器中**查找某个值是否存在的算法，存在返回true，否则返回false。

`binary_search(iterator_begin,iterator_end,value)`

参数分别为：起始迭代器、终止迭代器、指定值。例如：

```cpp
vector<int> vec = {1, 3, 5, 7, 9, 11, 13}; // 已排序容器
bool found = binary_search(vec.begin(), vec.end(), 5); //查找5
```

### 统计算法

#### `count()`

统计指定值出现的次数。

`count(iterator_begin,iterator_end,value)`

参数分别为：起始迭代器、终止迭代器、指定值。例如：

```cpp
int cnt = count(vec.begin(), vec.end(), 2); //统计2出现的次数。
```

#### `count_if()`

按条件统计元素个数

`count_if(iterator_begin,iterator_end,predicate)`

起始迭代器、终止迭代器、[一元谓词](#谓词)。例如：

```cpp
// 大于5的元素个数
int cnt = count_if(vec.begin(), vec.end(), [](int x) { return x > 5; });
```


### 排序算法



### 拷贝和替换函数



### 算术生成算法



### 集合算法
