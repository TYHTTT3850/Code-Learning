#include "Widget.h"
#include "ui_Widget.h"
#include <QString>
#include <QDebug>

Widget::Widget(QWidget *parent) :
    QWidget(parent), ui(new Ui::Widget) {
    ui->setupUi(this);
    connect(ui->num0, SIGNAL(clicked()), this, SLOT(inputNumber()));
    connect(ui->num1, SIGNAL(clicked()), this, SLOT(inputNumber()));
    connect(ui->confirm_button,SIGNAL(clicked()), ui->lineEdit, SIGNAL(returnPressed()));
    connect(ui->lineEdit,SIGNAL(returnPressed()),this,SLOT(outputText()));
}

Widget::~Widget() {
    delete ui;
}

void Widget:: inputNumber() {
    QPushButton* sender_num = dynamic_cast<QPushButton*>(sender()); //找出信号发射者
    QString text = ui->lineEdit->text(); //获取当前输入框中的内容
    ui->lineEdit->setText(text+sender_num->text());
}

void Widget:: outputText() {
    qDebug() << ui->lineEdit->text();
    ui->lineEdit->setText(""); //输出后清空输入框中的内容
}