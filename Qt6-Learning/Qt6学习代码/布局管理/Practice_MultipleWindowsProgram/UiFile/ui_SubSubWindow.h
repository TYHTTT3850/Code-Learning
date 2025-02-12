/********************************************************************************
** Form generated from reading UI file 'SubSubWindow.ui'
**
** Created by: Qt User Interface Compiler version 6.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SUBSUBWINDOW_H
#define UI_SUBSUBWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SubSubWindow
{
public:
    QLabel *label;

    void setupUi(QWidget *SubSubWindow)
    {
        if (SubSubWindow->objectName().isEmpty())
            SubSubWindow->setObjectName("SubSubWindow");
        SubSubWindow->resize(370, 218);
        label = new QLabel(SubSubWindow);
        label->setObjectName("label");
        label->setGeometry(QRect(110, 100, 151, 21));
        label->setAlignment(Qt::AlignmentFlag::AlignCenter);

        retranslateUi(SubSubWindow);

        QMetaObject::connectSlotsByName(SubSubWindow);
    } // setupUi

    void retranslateUi(QWidget *SubSubWindow)
    {
        SubSubWindow->setWindowTitle(QCoreApplication::translate("SubSubWindow", "SubSubWindow", nullptr));
        label->setText(QCoreApplication::translate("SubSubWindow", "\344\270\211\347\272\247\345\255\220\347\252\227\345\217\243", nullptr));
    } // retranslateUi

};

namespace Ui {
    class SubSubWindow: public Ui_SubSubWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SUBSUBWINDOW_H
