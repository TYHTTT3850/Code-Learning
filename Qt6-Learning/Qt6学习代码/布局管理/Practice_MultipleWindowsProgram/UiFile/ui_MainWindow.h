/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 6.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *verticalLayoutWidget;
    QVBoxLayout *verticalLayout;
    QLabel *main_window_label;
    QPushButton *open_subwindow_button;

    void setupUi(QWidget *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(550, 350);
        verticalLayoutWidget = new QWidget(MainWindow);
        verticalLayoutWidget->setObjectName("verticalLayoutWidget");
        verticalLayoutWidget->setGeometry(QRect(160, 120, 221, 101));
        verticalLayout = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout->setObjectName("verticalLayout");
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        main_window_label = new QLabel(verticalLayoutWidget);
        main_window_label->setObjectName("main_window_label");
        main_window_label->setAlignment(Qt::AlignmentFlag::AlignCenter);

        verticalLayout->addWidget(main_window_label);

        open_subwindow_button = new QPushButton(verticalLayoutWidget);
        open_subwindow_button->setObjectName("open_subwindow_button");
        open_subwindow_button->setAutoDefault(true);

        verticalLayout->addWidget(open_subwindow_button);


        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QWidget *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        main_window_label->setText(QCoreApplication::translate("MainWindow", "\344\270\273\347\252\227\345\217\243", nullptr));
        open_subwindow_button->setText(QCoreApplication::translate("MainWindow", "\346\230\276\347\244\272\344\272\214\347\272\247\345\255\220\347\252\227\345\217\243", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
