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
    void showSubWindow(); //自定义槽函数，按下按钮展示子窗口。

private:
    Ui::MainWindow *ui;
    SubWindow *sub_window; //主窗口除了主窗口的Ui外，还有子窗口。
};


#endif //MAINWINDOW_H
