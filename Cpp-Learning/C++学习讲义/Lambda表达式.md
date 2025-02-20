# Lambda表达式

Lambda 表达式的**一般形式**如下：

```cpp
[capture](parameter_list) -> return_type { function_body }
```

## 基本说明

### 各部分作用

| 部分                | 作用                     | 示例                |
| ------------------- | ------------------------ | ------------------- |
| `[capture]`         | **捕获外部变量**(可省略) | `[x, &y]`           |
| `(parameter_list)`  | **参数列表**(可选)       | `(int a, int b)`    |
| `-> return_type`    | **返回类型**(可省略)     | `-> int`            |
| `{ function_body }` | **函数体**               | `{ return a + b; }` |

### Lambda表达式生成闭包类型

当编译器遇到 Lambda 表达式时，会隐式生成一个**唯一的匿名类**(闭包类型)，该类包含以下内容：

1、重载的 `operator()` ：参数和返回类型与lambda的定义一致。

2、捕获的变量(如果有)：作为该类的成员变量。

**关键点：**

每个 Lambda 表达式都会生成一个全新的、不可显式命名的类型。

将 Lambda 表达式赋值给一个具体的变量时，就相当于进行了该类的实例化(创建对象)。

由于重载了 `operator()` ，可以实现类似于函数调用(类比 STL 的函数对象)。

由于 Lambda表达式 的类型是编译器生成的，无法直接通过类型名称显式声明。所以类型名称不可见，只能用 `auto` 声明变量，或者通过 `decltype` 推导类型。

建议优先使用 `auto` ，因为直接保留闭包类型，无额外开销。

## 基本示例

```cpp
#include <iostream>

int main() {
    auto add = [](int a, int b) -> int { return a + b; };
    std::cout << add(3, 4) << std::endl;  // 输出 7
    return 0;
}
```

上述示例中，将 Lambda 表达式赋值给了一个变量，从而让变量可以实现类似于函数的调用。

`-> int` **返回类型可省略**，编译器会自动推导。

## 省略 `-> return_type`

如果 `return` 语句能推导出返回类型，可以省略 `-> return_type`：

```cpp
auto add = [](int a, int b) { return a + b; };
```

## 无参数 Lambda

```cpp
auto greet = []() { std::cout << "Hello, Lambda!" << std::endl; };
greet();  // 输出 Hello, Lambda!
```

**无参数时** 仍然需要 `()`。

## 使用外部变量(捕获列表 `[]` )

```cpp
int x = 10;
auto lambda = [x](int y) { return x + y; };
std::cout << lambda(5) << std::endl;  // 输出 15
```

`[x]` **值捕获** `x` (Lambda 内部 `x` 是副本)。

## 按引用捕获 `&`

```cpp
int x = 10;
auto lambda = [&x]() { x += 5; };
lambda();
std::cout << x << std::endl;  // 输出 15
```

`[&x]` **引用捕获** `x` (修改外部变量)。

## 捕获所有变量

```cpp
int a = 5, b = 10;
auto lambda1 = [=]() { return a + b; };  // 以值捕获所有外部变量
auto lambda2 = [&]() { b += a; };        // 以引用捕获所有外部变量
```

| 捕获方式  | 作用                       |
| --------- | -------------------------- |
| `[=]`     | 以值捕获所有变量(不可修改) |
| `[&]`     | 以引用捕获所有变量(可修改) |
| `[x, &y]` | `x` 值捕获，`y` 引用捕获   |

## 总结

Lambda 表达式 **核心格式**：

```cpp
[capture](parameter_list) -> return_type { function_body }
```

💡 **一般会省略 `return_type`**，只写：

```cpp
[capture](parameter_list) { function_body }
```

**捕获变量**

- `[x]` **值捕获** `x` 。
- `[&x]` **引用捕获** `x` 。
- `[=]` **值捕获所有外部变量**。
- `[&]` **引用捕获所有外部变量**。

**应用场景**

- **用于 `sort()`、`find_if()` 等 STL 算法**。
- **作为回调函数**。
- **用于多线程编程**(`std::thread`)。

**Lambda 是 C++11 及以上的重要特性，非常常用！**