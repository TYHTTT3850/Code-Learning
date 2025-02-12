
#include "Widget.h"
#include "ui_Widget.h"


Widget::Widget(QWidget *parent) :
    QWidget(parent), ui(new Ui::Widget) {
    ui->setupUi(this);
    connect(ui->lineEdit,SIGNAL(returnPressed()),this,SLOT(enterIntoPlainTextEdit()));
}

Widget::~Widget() {
    delete ui;
}

void Widget::enterIntoPlainTextEdit() {
    ui->plainTextEdit->appendPlainText(ui->lineEdit->text());
    ui->lineEdit->clear();
}