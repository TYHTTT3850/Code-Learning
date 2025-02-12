//
// Created by Tyh11 on 25-1-19.
//

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QWidget>
#include "SubWindow.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QWidget {
Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void showSubWindow();

private:
    Ui::MainWindow *ui;
    SubWindow *sub_window;
};


#endif //MAINWINDOW_H
