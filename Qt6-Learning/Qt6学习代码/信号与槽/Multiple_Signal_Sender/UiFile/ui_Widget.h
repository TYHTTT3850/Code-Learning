/********************************************************************************
** Form generated from reading UI file 'Widget.ui'
**
** Created by: Qt User Interface Compiler version 6.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_H
#define UI_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QWidget *gridLayoutWidget;
    QGridLayout *gridLayout;
    QPushButton *sender_button1;
    QPushButton *sender_button2;
    QPushButton *sender_button3;
    QPushButton *sender_button4;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName("Widget");
        Widget->resize(519, 268);
        gridLayoutWidget = new QWidget(Widget);
        gridLayoutWidget->setObjectName("gridLayoutWidget");
        gridLayoutWidget->setGeometry(QRect(40, 90, 451, 112));
        gridLayout = new QGridLayout(gridLayoutWidget);
        gridLayout->setObjectName("gridLayout");
        gridLayout->setContentsMargins(0, 0, 0, 0);
        sender_button1 = new QPushButton(gridLayoutWidget);
        sender_button1->setObjectName("sender_button1");
        sender_button1->setFlat(false);

        gridLayout->addWidget(sender_button1, 0, 0, 1, 1);

        sender_button2 = new QPushButton(gridLayoutWidget);
        sender_button2->setObjectName("sender_button2");

        gridLayout->addWidget(sender_button2, 0, 1, 1, 1);

        sender_button3 = new QPushButton(gridLayoutWidget);
        sender_button3->setObjectName("sender_button3");

        gridLayout->addWidget(sender_button3, 1, 0, 1, 1);

        sender_button4 = new QPushButton(gridLayoutWidget);
        sender_button4->setObjectName("sender_button4");

        gridLayout->addWidget(sender_button4, 1, 1, 1, 1);


        retranslateUi(Widget);

        sender_button1->setDefault(true);
        sender_button2->setDefault(true);
        sender_button3->setDefault(true);
        sender_button4->setDefault(true);


        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QCoreApplication::translate("Widget", "Widget", nullptr));
        sender_button1->setText(QCoreApplication::translate("Widget", "signal1", nullptr));
        sender_button2->setText(QCoreApplication::translate("Widget", "signal2", nullptr));
        sender_button3->setText(QCoreApplication::translate("Widget", "signal3", nullptr));
        sender_button4->setText(QCoreApplication::translate("Widget", "signal4", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
