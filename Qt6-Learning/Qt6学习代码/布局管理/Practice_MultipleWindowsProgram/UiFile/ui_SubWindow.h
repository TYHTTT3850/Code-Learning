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
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SubWindow
{
public:
    QWidget *verticalLayoutWidget;
    QVBoxLayout *verticalLayout;
    QLabel *sub_window_label;
    QPushButton *show_subsubwindow_button;

    void setupUi(QWidget *SubWindow)
    {
        if (SubWindow->objectName().isEmpty())
            SubWindow->setObjectName("SubWindow");
        SubWindow->resize(400, 300);
        verticalLayoutWidget = new QWidget(SubWindow);
        verticalLayoutWidget->setObjectName("verticalLayoutWidget");
        verticalLayoutWidget->setGeometry(QRect(120, 100, 160, 80));
        verticalLayout = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout->setObjectName("verticalLayout");
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        sub_window_label = new QLabel(verticalLayoutWidget);
        sub_window_label->setObjectName("sub_window_label");
        sub_window_label->setAlignment(Qt::AlignmentFlag::AlignCenter);

        verticalLayout->addWidget(sub_window_label);

        show_subsubwindow_button = new QPushButton(verticalLayoutWidget);
        show_subsubwindow_button->setObjectName("show_subsubwindow_button");

        verticalLayout->addWidget(show_subsubwindow_button);


        retranslateUi(SubWindow);

        QMetaObject::connectSlotsByName(SubWindow);
    } // setupUi

    void retranslateUi(QWidget *SubWindow)
    {
        SubWindow->setWindowTitle(QCoreApplication::translate("SubWindow", "SubWindow", nullptr));
        sub_window_label->setText(QCoreApplication::translate("SubWindow", "\344\272\214\347\272\247\347\252\227\345\217\243", nullptr));
        show_subsubwindow_button->setText(QCoreApplication::translate("SubWindow", "\346\230\276\347\244\272\344\270\211\347\272\247\347\252\227\345\217\243", nullptr));
    } // retranslateUi

};

namespace Ui {
    class SubWindow: public Ui_SubWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SUBWINDOW_H
