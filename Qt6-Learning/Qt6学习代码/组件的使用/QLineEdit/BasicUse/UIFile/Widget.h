//
// Created by Tyh11 on 25-1-21.
//

#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>


QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget {
Q_OBJECT

public:
    explicit Widget(QWidget *parent = nullptr);
    ~Widget() override;

private slots:
    void outputTextChanged();

private:
    Ui::Widget *ui;
};


#endif //WIDGET_H
