// QPlainEdit 是一个多行纯文本编辑控件，可以用于输入和显示多行纯文本

#include <QApplication>
#include "UIFile/Widget.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    Widget w;
    w.show();
    return QApplication::exec();
}
