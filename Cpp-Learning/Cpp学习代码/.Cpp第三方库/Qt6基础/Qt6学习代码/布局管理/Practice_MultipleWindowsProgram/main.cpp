// 设计一个具有三级窗口的应用程序

#include <QApplication>
#include "UiFile/MainWindow.h"
#include "UiFile/ui_MainWindow.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    MainWindow main_window;
    main_window.show();
    return QApplication::exec();
}