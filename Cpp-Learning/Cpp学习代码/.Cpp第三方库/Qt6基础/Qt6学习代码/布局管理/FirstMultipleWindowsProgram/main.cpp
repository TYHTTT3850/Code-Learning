#include <QApplication>
#include "UiFile/MainWindow.h"
#include "UiFile/ui_MainWindow.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    MainWindow mw;
    mw.show();
    return QApplication::exec();
}