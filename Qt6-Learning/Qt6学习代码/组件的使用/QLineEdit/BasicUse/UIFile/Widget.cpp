
#include "Widget.h"
#include "ui_Widget.h"
#include <QDebug>
#include <QString>


Widget::Widget(QWidget *parent) :
    QWidget(parent), ui(new Ui::Widget) {
    ui->setupUi(this);
    connect(ui->lineEdit, SIGNAL(returnPressed()), this, SLOT(outputTextChanged()));
}

Widget::~Widget() {
    delete ui;
}

void Widget::outputTextChanged() {
    QString text = ui->lineEdit->text();
    qDebug() << text;
}