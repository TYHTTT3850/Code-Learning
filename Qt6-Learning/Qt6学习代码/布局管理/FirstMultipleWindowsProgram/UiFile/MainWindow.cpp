//
// Created by Tyh11 on 25-1-19.
//

// You may need to build the project (run Qt uic code generator) to get "ui_MainWindow.h" resolved

#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "SubWindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QWidget(parent), ui(new Ui::MainWindow), sub_window(nullptr){
    ui->setupUi(this);
    connect(ui->ShowSubWindowButton, SIGNAL(clicked()), this, SLOT(showSubWindow()));
}

MainWindow::~MainWindow() {
    delete ui;
    delete sub_window;
}

//显示子窗口的槽函数的实现
void MainWindow::showSubWindow() {
    if (!sub_window) { //如果子窗口未创建，则创建一个子窗口
        sub_window = new SubWindow(); //不把子窗口嵌入在主窗口中，而是新开辟一个窗口。
    }
    sub_window->show(); //显示子窗口
    sub_window->raise(); //子窗口保持在最前面
    sub_window->activateWindow(); //将子窗口激活并使其成为当前活动窗口
}