//此文件是对应于 Widget.h 的源程序文件

#include "Widget.h"
#include "ui_Widget.h"


Widget::Widget(QWidget *parent) :
    QWidget(parent), ui(new Ui::Widget) {
    ui->setupUi(this);
}

Widget::~Widget() {
    delete ui;
}
