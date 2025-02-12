## Qt6 工程介绍

`Qt Widgets Application` ：创建图形化用户界面的传统桌面程序。

`Qt Console Application` ：创建没有图形用户界面的控制台应用程序。

一般使用`Qt Widgets Application`。

注意：创建的工程路径中不能带有任何中文字符！！！

## Qt6 Ui 类

创建 Ui 类时主要有三种基类。

**1、 `QWidget`** ：所有界面组件类的基类，QWidget可以作为独立的窗口，就是一个空白的窗口。

**2、 `QMainWindow`** ：主窗口类，具有菜单栏，工具栏，状态栏。

**3、 `QDialog`** ：对话框类，具有对话框的显示效果，例如没有最大化按钮。

## 信号与槽

信号与槽是 Qt 编程的基础，也是 Qt 的一大创新。有了信号与槽的编程机制，在 Qt 中处理界面上各个组件的交互操作就变得比较直观和简单。

信号（signal）是在特定情况下被发射的通知。信号可以看作特殊的函数。信号函数必须是无返回值的函数，但是可以有输入参数。信号函数无须实现，而只需在某些条件下被发射。用 Qt 中的关键字 `emit` 发射信号。

槽（slot）是对信号进行响应的函数，槽就是函数，所以也称为槽函数。

### `connect()` 函数

信号与槽关联是用函数 `QObject::connect()` 实现的，`connect()` 有一种成员函数形式，还有多种静态函数形式。一般使用静态函数形式。

静态函数 `QObject::connect()` 有多种参数形式。

**1、第一种参数形式的静态函数 `connect()`**

函数原型为：

```cpp

static QMetaObject::Connection QObject::connect(const QObject *sender, const char *signal, const QObject *receiver, const char *method, Qt::ConnectionType type = Qt::AutoConnection);

```

使用这种参数形式的 `connect()` 函数进行信号与槽函数的连接时，一般用法如下：

```cpp

QObject::connect(sender, SIGNAL(signal()), receiver, SLOT(slot()));

```

`connect()` 是 QObject 类的一个静态函数，而 QObject 是大部分 Qt 类的基类，实际上，在类中调用静态函数时可
以忽略前面的限定符部分，所以可以直接写为：

```cpp

connect(sender, SIGNAL(signal()), receiver, SLOT(slot()));

```

其中，`sender` 是发射信号的对象的名称，`signal()` 是属于 `sender` 的信号，有参数时还需要指明各参数类型，但不用指明参数名称；`receiver` 是接收信号的对象的名称，`slot()` 是属于 `receiver` 的槽函数，有参数时还需要指明各参数类型，但不用指明参数名称。`SIGNAL` 和 `SLOT` 是 Qt 的宏，分别用于指明信号和槽函数。

**2、第二种参数形式的静态函数 `connect()`**

函数原型为：

```cpp

static QMetaObject::Connection QObject::connect(const QObject *sender, const QMetaMethod &signal, const QObject *receiver, const QMetaMethod &method, Qt::ConnectionType type = Qt::AutoConnection);

```

对于信号名称是唯一的信号函数，即不存在参数不同的其他同名的信号，可以使用这种函数指针形式进行关联。如下所示：

```cpp

connect(sender, &signal, receiver, &slot);

```

同样的，`sender` 是发射信号的对象的名称，`signal` 是属于 `sender` 的信号；`receiver` 是接收信号的对象的名称，`slot` 是属于 `receiver` 的槽函数。

#### 连接方式 `Qt::ConnectionType`

不管是哪种参数形式的静态 `connect()` 函数，最后都有一个参数 `type` ，它是枚举类型 `Qt::ConnectionType` ，
默认值为 `Qt::AutoConnection` 。枚举类型 `Qt::ConnectionType` 表示信号与槽的关联方式， 有以下几种取值：

**1、 `Qt::AutoConnection` (默认值)**：如果信号的接收者与发射者在同一个线程中，就使用 `Qt::DirectConnection` 方式，否则使用 `Qt::QueuedConnection` 方式，在信号发射时自动确定关联方式。

**2、`Qt::DirectConnection`**：信号被发射时槽函数立即运行，槽函数与信号在同一个线程中。

**3、`Qt::QueuedConnection`** ：在事件循环回到接收者线程后运行槽函数，槽函数与信号在不同的线程中。

**4、`Qt::BlockingQueuedConnection`** ：与 `Qt::QueuedConnection` 相似，区别是信号线程会阻塞，
直到槽函数运行完毕。当信号与槽函数在同一个线程中时绝对不能使用这种方式，否则会造成死锁。

#### 作为 QObject 成员函数的 `connect()` 函数 

函数原型定义如下：

```cpp

QMetaObject::Connection QObject::connect(const QObject *sender, const char *signal, const char *method, Qt::ConnectionType type = Qt::AutoConnection);

```

这个函数里没有表示接收者的参数，接收者就是对象自身。例如，使用静态函数 `connect()` 设置连接的一条语句如下：

```cpp

connect(sender, SIGNAL(signal()), this, SLOT(slot()));

```
如果使用成员函数 `connect()` ，就可以写成如下的语句：

```cpp

this->connect(sender, SIGNAL(signal()), SLOT(slot()));

```

### 信号与槽连接的规则

**1、一个信号可以连接多个槽函数。当一个信号与多个槽函数关联时，槽函数按照建立连接时的顺序依次运行。**

**2、多个信号可以连接同一个槽函数。**

**3、一个信号可以连接另一个信号。这样，当发射一个信号时，也会发射另一个信号，以实现某些特殊的功能。例如：**

```cpp

connect(sender, SIGNAL(signal_1()), receiver, SIGNAL(signal_2()));

```

**4、严格的情况下，信号与槽的参数个数和类型需要一致，至少信号的参数不能少于槽的参数。如果参数不匹配，会出现编译错误或运行错误。**

**5、在使用信号与槽的类中，必须在类的定义中插入宏Q_OBJECT。**

**6、当一个信号被发射时，与其关联的槽函数通常被立即运行，就像正常调用函数一样。只有当信号关联的所有槽函数运行完毕后，才运行发射信号处后面的代码。**





