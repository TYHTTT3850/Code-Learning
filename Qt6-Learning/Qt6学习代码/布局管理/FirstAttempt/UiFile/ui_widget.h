/********************************************************************************
** Form generated from reading UI file 'widget.ui'
**
** Created by: Qt User Interface Compiler version 6.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_H
#define UI_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QLabel *WelcomeLabel;
    QPushButton *CloseButton;
    QPushButton *HideButton;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName("Widget");
        Widget->resize(387, 287);
        WelcomeLabel = new QLabel(Widget);
        WelcomeLabel->setObjectName("WelcomeLabel");
        WelcomeLabel->setGeometry(QRect(130, 130, 131, 20));
        QFont font;
        font.setFamilies({QString::fromUtf8("Microsoft YaHei")});
        font.setPointSize(12);
        WelcomeLabel->setFont(font);
        WelcomeLabel->setAlignment(Qt::AlignmentFlag::AlignCenter);
        CloseButton = new QPushButton(Widget);
        CloseButton->setObjectName("CloseButton");
        CloseButton->setGeometry(QRect(270, 230, 75, 23));
        CloseButton->setAutoDefault(true);
        HideButton = new QPushButton(Widget);
        HideButton->setObjectName("HideButton");
        HideButton->setGeometry(QRect(190, 230, 75, 23));
        HideButton->setAutoDefault(true);

        retranslateUi(Widget);
        QObject::connect(CloseButton, &QPushButton::clicked, Widget, qOverload<>(&QWidget::close));
        QObject::connect(HideButton, &QPushButton::clicked, Widget, qOverload<>(&QWidget::showMinimized));

        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QCoreApplication::translate("Widget", "FirstAttempt", nullptr));
        WelcomeLabel->setText(QCoreApplication::translate("Widget", "Welcome to Qt6", nullptr));
        CloseButton->setText(QCoreApplication::translate("Widget", "\345\205\263\351\227\255", nullptr));
        HideButton->setText(QCoreApplication::translate("Widget", "\346\234\200\345\260\217\345\214\226", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
