//
// Created by Tyh11 on 25-1-19.
//

#ifndef SUBWINDOW_H
#define SUBWINDOW_H

#include <QWidget>


QT_BEGIN_NAMESPACE
namespace Ui { class SubWindow; }
QT_END_NAMESPACE

class SubWindow : public QWidget {
Q_OBJECT

public:
    explicit SubWindow(QWidget *parent = nullptr);
    ~SubWindow() override;

private:
    Ui::SubWindow *ui;
};


#endif //SUBWINDOW_H
