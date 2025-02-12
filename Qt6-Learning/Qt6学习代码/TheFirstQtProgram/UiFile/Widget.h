#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget> //包含 QWidget 窗口类头文件

//声明一个名字空间，包含一个类 Widget 。这个类不是此文件中定义的，而是在 ui_Widget.h 文件中定义的。
QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget {
Q_OBJECT // Q_Object 为 Qt 中的一个宏，该宏允许类使用信号和槽的机制

public:
    explicit Widget(QWidget *parent = nullptr);
    ~Widget() override;

private:
    //称 Ui::myWidget 为窗口Ui类
    Ui::Widget *ui; //窗口 Ui 类指针，指向可视化设计的窗口界面，要访问界面上的组件，就需要通过该指针实现
};


#endif //WIDGET_H
