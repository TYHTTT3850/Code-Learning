//
// Created by Tyh11 on 25-1-19.
//

// You may need to build the project (run Qt uic code generator) to get "ui_SubWindow.h" resolved

#include "SubWindow.h"
#include "ui_SubWindow.h"



SubWindow::SubWindow(QWidget *parent) :
    QWidget(parent), ui(new Ui::SubWindow), sub_sub_window(nullptr){
    ui->setupUi(this);
    connect(ui->show_subsubwindow_button, SIGNAL(clicked()), this, SLOT(showSubSubWindow()));
}

SubWindow::~SubWindow() {
    delete ui;
    delete sub_sub_window;
}

void SubWindow::showSubSubWindow() {
    if (sub_sub_window == nullptr) {
        sub_sub_window = new SubSubWindow();
    }
    sub_sub_window->show();
    sub_sub_window->raise();
    sub_sub_window->activateWindow();
}