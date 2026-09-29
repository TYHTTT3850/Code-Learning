#include <QApplication> //包含应用程序类的头文件
#include "UiFile/Widget.h"
#include "UiFile/ui_Widget.h"

//文件ui_widget.h 里封装界面的类是Ui_Widget。

//再定义一个类 Widget 从 Ui_Widget 继承而来，并将其定义在名字空间 Ui 里，这样 Ui::Widget 与文件 widget.h 里的类 Widget 同名，但是用名字空间区分开。

//在文件 ui_widget.h 里有一个名字空间Ui，里面有一个类Widget，记作 Ui::Widget，它是窗口 UI 类。

//文件 widget.h 里的类 Widget 是完整的窗口类。

//main：程序的入口。argc：命令行变量的数量。argv：命令行变量的数组。
int main(int argc, char *argv[]) {
    //a：应用程序对象，在Qt中，应用程序对象有且仅有一个。
    QApplication a(argc, argv);

    //创建窗口对象
    Widget w;

    //窗口对象默认不会显示，必须调用show方法
    w.show();

    //让应用程序对象进入事件循环，从而程序不会一闪而过
    return QApplication::exec();
}