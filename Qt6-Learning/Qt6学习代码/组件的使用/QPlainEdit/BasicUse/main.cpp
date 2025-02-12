// QPlainEdit 是一个多行纯文本编辑控件，可以用于输入和显示多行纯文本
// 本例设计了从 QLineEdit 输入，QPlainEdit 显示输入的历史记录的功能

#include <QApplication>
#include "UiFile/Widget.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    Widget w;
    w.show();
    return QApplication::exec();
}