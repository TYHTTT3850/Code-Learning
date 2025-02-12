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
    QLabel *MainWindowInformation;
    QPushButton *ShowSubWindowButton;

    void setupUi(QWidget *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(487, 384);
        verticalLayoutWidget = new QWidget(MainWindow);
        verticalLayoutWidget->setObjectName("verticalLayoutWidget");
        verticalLayoutWidget->setGeometry(QRect(160, 150, 191, 81));
        verticalLayout = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout->setObjectName("verticalLayout");
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        MainWindowInformation = new QLabel(verticalLayoutWidget);
        MainWindowInformation->setObjectName("MainWindowInformation");
        MainWindowInformation->setAlignment(Qt::AlignmentFlag::AlignCenter);

        verticalLayout->addWidget(MainWindowInformation);

        ShowSubWindowButton = new QPushButton(verticalLayoutWidget);
        ShowSubWindowButton->setObjectName("ShowSubWindowButton");

        verticalLayout->addWidget(ShowSubWindowButton);


        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QWidget *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        MainWindowInformation->setText(QCoreApplication::translate("MainWindow", "\344\270\273\347\252\227\345\217\243", nullptr));
        ShowSubWindowButton->setText(QCoreApplication::translate("MainWindow", "\346\230\276\347\244\272\345\255\220\347\252\227\345\217\243", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
