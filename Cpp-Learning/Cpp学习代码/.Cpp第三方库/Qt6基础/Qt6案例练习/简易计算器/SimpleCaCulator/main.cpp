#include <QApplication>
#include "UiFile/Caculator.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    Caculator c;
    c.show();
    return QApplication::exec();
}