#ifndef WIDGET_H
#define WIDGET_H

#include <QPushButton>
#include <QWidget>
#include <QDebug>

class Widget : public QWidget {
    Q_OBJECT
public:
    QPushButton *sender_button;
    Widget(QWidget *parent = nullptr):sender_button(new QPushButton("sender_button",this)) {
        sender_button->setObjectName("sender_button");
        connect(sender_button,&QPushButton::clicked,this,&onButtonClicked);
    }
    private slots:
        void onButtonClicked() {
        QPushButton *button = qobject_cast<QPushButton*>(sender()); //利用sender()获取信号发射者。
        //qobject_cast是专门为 Qt 对象设计的类型转换的函数模板

        //qDebug()是专门为 Qt 设计的输出调试信息的方法。
        qDebug() << "Sender:" << button->objectName();
    }

};

#endif //WIDGET_H
