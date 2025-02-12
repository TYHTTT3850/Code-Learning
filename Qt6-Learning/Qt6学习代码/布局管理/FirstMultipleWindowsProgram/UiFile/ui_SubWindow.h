/********************************************************************************
** Form generated from reading UI file 'SubWindow.ui'
**
** Created by: Qt User Interface Compiler version 6.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SUBWINDOW_H
#define UI_SUBWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SubWindow
{
public:
    QLabel *SubWindowInformation;

    void setupUi(QWidget *SubWindow)
    {
        if (SubWindow->objectName().isEmpty())
            SubWindow->setObjectName("SubWindow");
        SubWindow->resize(400, 300);
        SubWindowInformation = new QLabel(SubWindow);
        SubWindowInformation->setObjectName("SubWindowInformation");
        SubWindowInformation->setGeometry(QRect(140, 110, 121, 51));
        SubWindowInformation->setAlignment(Qt::AlignmentFlag::AlignCenter);

        retranslateUi(SubWindow);

        QMetaObject::connectSlotsByName(SubWindow);
    } // setupUi

    void retranslateUi(QWidget *SubWindow)
    {
        SubWindow->setWindowTitle(QCoreApplication::translate("SubWindow", "SubWindow", nullptr));
        SubWindowInformation->setText(QCoreApplication::translate("SubWindow", "\345\255\220\347\252\227\345\217\243", nullptr));
    } // retranslateUi

};

namespace Ui {
    class SubWindow: public Ui_SubWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SUBWINDOW_H
