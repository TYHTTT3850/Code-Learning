//
// Created by Tyh11 on 25-1-19.
//

#ifndef SUBSUBWINDOW_H
#define SUBSUBWINDOW_H

#include <QWidget>


QT_BEGIN_NAMESPACE
namespace Ui { class SubSubWindow; }
QT_END_NAMESPACE

class SubSubWindow : public QWidget {
Q_OBJECT

public:
    explicit SubSubWindow(QWidget *parent = nullptr);
    ~SubSubWindow() override;

private:
    Ui::SubSubWindow *ui;
};


#endif //SUBSUBWINDOW_H
