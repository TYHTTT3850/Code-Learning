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
    connect(ui->open_subwindow_button, SIGNAL(clicked()), this, SLOT(showSubWindow()));
}

MainWindow::~MainWindow() {
    delete ui;
    delete sub_window;
}

void MainWindow::showSubWindow() {
    if (sub_window == nullptr) {
        sub_window = new SubWindow();
    }
    sub_window->show();
    sub_window->raise();
    sub_window->activateWindow();
}
