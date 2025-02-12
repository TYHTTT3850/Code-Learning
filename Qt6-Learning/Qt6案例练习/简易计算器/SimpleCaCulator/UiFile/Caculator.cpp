
#include <QString>
#include "Caculator.h"
#include "ui_Caculator.h"
#include "exprtk.hpp" //表达式解析库，下载地址https://github.com/ArashPartow/exprtk.git，下载后直接放入项目中即可使用
//加入 exprtk.hpp 后编译的文件太大，故使用 MSVC 编译，同时在 CMake 中加入 /bigobj 指令
#include <iostream>


Caculator::Caculator(QWidget *parent) :
    QWidget(parent), ui(new Ui::Caculator) {
    ui->setupUi(this);

    //设置数字和符号按钮的信号和槽
    connect(ui->num0_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->num1_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->num2_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->num3_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->num4_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->num5_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->num6_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->num7_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->num8_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->num9_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->plus_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->minus_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->multiply_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->divide_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->left_bracket_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->right_bracket_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));
    connect(ui->dot_button, SIGNAL(clicked()), this, SLOT(enterIntoLineEdit()));

    //单独设置百分号的信号和槽函数
    connect(ui->percent_button, SIGNAL(clicked()), this, SLOT(percentageIntoLineEdit()));

    //设置exit按钮的信号和槽函数
    connect(ui->exit_button, SIGNAL(clicked()), this, SLOT(close()));

    //设置ac按钮的信号和槽函数
    connect(ui->ac_button, SIGNAL(clicked()), ui->lineEdit, SLOT(clear()));

    //设置clear_history按钮的信号和槽函数
    connect(ui->clear_history_button, SIGNAL(clicked()), ui->plainTextEdit, SLOT(clear()));

    //设置backspace按钮的信号和槽函数
    connect(ui->backspace_button, SIGNAL(clicked()), this, SLOT(backspaceIntoLineEdit()));

    //连接等号按钮的信号和槽函数
    connect(ui->equal_button, SIGNAL(clicked()), this, SLOT(caculateResult()));
}

Caculator::~Caculator() {
    delete ui;
}

void Caculator::enterIntoLineEdit() {
    QPushButton* sender_button = qobject_cast<QPushButton*>(sender()); //获取信号的发射者
    ui->lineEdit->setText(ui->lineEdit->text()+sender_button->text());
}

void Caculator::backspaceIntoLineEdit() {
    QString text =ui->lineEdit->text();
  ui->lineEdit->setText(ui->lineEdit->text().remove(-1,1)); //-1表示末尾，1表示删除一个字符
}

void Caculator::percentageIntoLineEdit() {
    ui->lineEdit->setText(ui->lineEdit->text()+"/100");
}

void Caculator::caculateResult() {
    std::string expression = ui->lineEdit->text().toStdString(); //获取C++标准字符串以用于数学表达式解析

    //创建表达式对象和解析器对象
    exprtk::expression<double> expr;
    exprtk::parser<double> parser;

    if (parser.compile(expression, expr)) {
        ui->plainTextEdit->appendPlainText(ui->lineEdit->text()+"="+QString::number(expr.value())); //结果使用 QString::number() 转换为 QString

        ui->lineEdit->setText("");
    }
    else {
        ui->lineEdit->setText("error");
    }
}
