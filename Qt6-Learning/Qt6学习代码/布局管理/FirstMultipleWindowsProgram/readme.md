## 窗口的创建

首先使用 `Qt Designer` 创建好所有的窗口及其组件，然后确定窗口间的主次关系。

## 建立窗口间的连接

确定好主次关系，建立主窗口和子窗口的连接主要在主窗口的头文件和源文件上操作。实际设计中，窗口可能会有多级关系，所以以最简单的二级关系，即一个主窗口和一个子窗口为例，多级窗口关系无非就是简单的二级关系的多层嵌套或组合。

把主窗口设为 `MainWindow` ，次窗口设为 `SubWindow` 。他们所属的文件如下所示：

`MainWindow` ：`MainWindow.cpp`，`MainWindow.h`，`ui_MainWindow.h`。

`SubWindow` ：`SubWindow.cpp`，`SubWindow.h`，`ui_SubWindow.h`。

建立两个窗口间的连接主要是对 `MainWindow.cpp` 和 `MainWindow.h` 操作。主要分为以下几个步骤：1、将子窗口类组合进主窗口类，也就是说，主窗口类的成员变量除了其自身的窗口Ui类之外，还有子窗口类。2、利用信号与槽的机制建立主窗口和子窗口之间的连接。

### 子窗口类组合进主窗口类

在 `MainWindow.h` 头文件中给出类主窗口类的定义，其私有成员变量只有其自身的窗口Ui类。如下所示：

```cpp

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QWidget>
#include "SubWindow.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QWidget {
Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private:
    Ui::MainWindow *ui;
};


#endif //MAINWINDOW_H

```

现将子窗口类组合进去：包含子窗口类的头文件 `SubWindow.h` ，然后在私有成员变量中加入SubWindow类的对象。代码如下：

```cpp

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QWidget>
#include "SubWindow.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QWidget {
Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private:
    Ui::MainWindow *ui;
    SubWindow *sub_window; //主窗口除了其窗口Ui类外，还有子窗口类。
};


#endif //MAINWINDOW_H

```

### 编辑信号与槽

主窗口经过某个事件发出一个信号，此信号会触发一个显示子窗口的槽函数 `showSubWindow()` 。主窗口作为应用的控制中心，负责统一管理和调度子窗口的创建与显示，所以在 `MainWindow`类 中定义这个槽函数。如下所示：

```cpp
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QWidget>
#include "SubWindow.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QWidget {
Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void showSubWindow(); //自定义槽函数，按下按钮展示子窗口。

private:
    Ui::MainWindow *ui;
    SubWindow *sub_window; //主窗口除了主窗口的Ui外，还有子窗口。
};


#endif //MAINWINDOW_H


```

至此， `MainWindow.h` 的定义补充已经完成。需要在 `MainWindow.cpp` 中完成具体实现。

## `MainWindow.cpp` 的具体实现

在没有建立主窗口和子窗口的连接之前，`MainWindow.cpp` 的代码如下：

```cpp

#include "MainWindow.h"
#include "ui_MainWindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QWidget(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);

MainWindow::~MainWindow() {
    delete ui;
}

```

由于将 `SubWindow` 类组合进了 `MainWindow` 类，所以其构造函数的初始化列表需要将组合进来的 `SubWindow` 类对象也一并初始化了，同时析构函数也要同时把新组合来的 `SubWindow` 类对象一并删除。代码如下：

```cpp

#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "SubWindow.h" //包含 SubWindow 类的头文件

MainWindow::MainWindow(QWidget *parent) :
    QWidget(parent), ui(new Ui::MainWindow), sub_window(nullptr){
    ui->setupUi(this);
}
//

MainWindow::~MainWindow() {
    delete ui;
    delete sub_window;
}

```

在初始化列表中写 `sub_window(nullptr)` 是为了不设置父窗口，使得子窗口成为独立窗口，而不是嵌套在主窗口内部。这样做可以确保子窗口在显示时不依赖于主窗口的布局，并能够作为一个独立的窗口弹出。

然后将信号与槽函数进行连接，使用 `connect()` 方法，这里以鼠标的点击信号 `clicked()` 为例，点击按钮后，展示子窗口，代码如下：

```cpp

MainWindow::MainWindow(QWidget *parent) :
    QWidget(parent), ui(new Ui::MainWindow), sub_window(nullptr){
    ui->setupUi(this);
    connect(ui->ShowSubWindowButton, SIGNAL(clicked()), this, SLOT(showSubWindow())); //将信号与槽函数连接
}

```

最后就是实现能够显示子窗口的槽函数代码如下：

```cpp

void MainWindow::showSubWindow() {
    if (!sub_window) { //如果子窗口未创建，则创建一个子窗口
        sub_window = new SubWindow(); //不把子窗口嵌入在主窗口中，而是新开辟一个窗口。
    }
    sub_window->show(); //显示子窗口
    sub_window->raise(); //子窗口保持在最前面
    sub_window->activateWindow(); //将子窗口激活并使其成为当前活动窗口
}

```
最终，补充后的 `MainWindow.cpp` 完整如下所示：

```cpp

#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "SubWindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QWidget(parent), ui(new Ui::MainWindow), sub_window(nullptr){
    ui->setupUi(this);
    connect(ui->ShowSubWindowButton, SIGNAL(clicked()), this, SLOT(showSubWindow()));
}

MainWindow::~MainWindow() {
    delete ui;
    delete sub_window;
}

//显示子窗口的槽函数的实现
void MainWindow::showSubWindow() {
    if (!sub_window) { //如果子窗口未创建，则创建一个子窗口
        sub_window = new SubWindow(); //不把子窗口嵌入在主窗口中，而是新开辟一个窗口。
    }
    sub_window->show(); //显示子窗口
    sub_window->raise(); //子窗口保持在最前面
    sub_window->activateWindow(); //将子窗口激活并使其成为当前活动窗口
}

```

### `MainWindow.cpp` 的另一种写法

上述写法在 `MainWindow` 的构造函数中先将子窗口初始化为空指针(因为不一定会用到子窗口)，当然也可以直接在构造时就直接初始化子窗口，那么这样的 `MainWindow.cpp` 的实现如下所示：

```cpp

#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "SubWindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QWidget(parent), ui(new Ui::MainWindow), sub_window(new SubWindow){ //直接初始化子窗口
    ui->setupUi(this);
    connect(ui->ShowSubWindowButton, SIGNAL(clicked()), this, SLOT(showSubWindow()));
}

MainWindow::~MainWindow() {
    delete ui;
    delete sub_window;
}

//显示子窗口的槽函数的实现
void MainWindow::showSubWindow() {
    //不再需要判断子窗口是否创建
    sub_window->show(); //显示子窗口
    sub_window->raise(); //子窗口保持在最前面
    sub_window->activateWindow(); //将子窗口激活并使其成为当前活动窗口
}

```
