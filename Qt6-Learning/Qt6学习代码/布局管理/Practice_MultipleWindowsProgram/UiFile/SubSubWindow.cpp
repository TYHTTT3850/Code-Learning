//
// Created by Tyh11 on 25-1-19.
//

// You may need to build the project (run Qt uic code generator) to get "ui_SubSubWindow.h" resolved

#include "SubSubWindow.h"
#include "ui_SubSubWindow.h"


SubSubWindow::SubSubWindow(QWidget *parent) :
    QWidget(parent), ui(new Ui::SubSubWindow) {
    ui->setupUi(this);
}

SubSubWindow::~SubSubWindow() {
    delete ui;
}
