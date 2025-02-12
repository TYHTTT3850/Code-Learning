//
// Created by Tyh11 on 25-1-19.
//

// You may need to build the project (run Qt uic code generator) to get "ui_SubWindow.h" resolved

#include "SubWindow.h"
#include "ui_SubWindow.h"


SubWindow::SubWindow(QWidget *parent) :
    QWidget(parent), ui(new Ui::SubWindow) {
    ui->setupUi(this);
}

SubWindow::~SubWindow() {
    delete ui;
}
