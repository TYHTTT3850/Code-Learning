/********************************************************************************
** Form generated from reading UI file 'Caculator.ui'
**
** Created by: Qt User Interface Compiler version 6.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CACULATOR_H
#define UI_CACULATOR_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Caculator
{
public:
    QVBoxLayout *verticalLayout_2;
    QVBoxLayout *verticalLayout;
    QPlainTextEdit *plainTextEdit;
    QLineEdit *lineEdit;
    QGridLayout *gridLayout;
    QPushButton *minus_button;
    QPushButton *num1_button;
    QPushButton *num7_button;
    QPushButton *percent_button;
    QPushButton *plus_button;
    QPushButton *num5_button;
    QPushButton *ac_button;
    QPushButton *num6_button;
    QPushButton *num2_button;
    QPushButton *num3_button;
    QPushButton *num9_button;
    QPushButton *num4_button;
    QPushButton *num0_button;
    QPushButton *num8_button;
    QPushButton *divide_button;
    QPushButton *equal_button;
    QPushButton *clear_history_button;
    QPushButton *dot_button;
    QPushButton *multiply_button;
    QPushButton *backspace_button;
    QPushButton *right_bracket_button;
    QPushButton *left_bracket_button;
    QPushButton *exit_button;

    void setupUi(QWidget *Caculator)
    {
        if (Caculator->objectName().isEmpty())
            Caculator->setObjectName("Caculator");
        Caculator->resize(406, 460);
        verticalLayout_2 = new QVBoxLayout(Caculator);
        verticalLayout_2->setObjectName("verticalLayout_2");
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName("verticalLayout");
        plainTextEdit = new QPlainTextEdit(Caculator);
        plainTextEdit->setObjectName("plainTextEdit");
        QSizePolicy sizePolicy(QSizePolicy::Policy::Preferred, QSizePolicy::Policy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(plainTextEdit->sizePolicy().hasHeightForWidth());
        plainTextEdit->setSizePolicy(sizePolicy);
        plainTextEdit->setReadOnly(true);

        verticalLayout->addWidget(plainTextEdit);

        lineEdit = new QLineEdit(Caculator);
        lineEdit->setObjectName("lineEdit");
        sizePolicy.setHeightForWidth(lineEdit->sizePolicy().hasHeightForWidth());
        lineEdit->setSizePolicy(sizePolicy);
        lineEdit->setReadOnly(false);

        verticalLayout->addWidget(lineEdit);


        verticalLayout_2->addLayout(verticalLayout);

        gridLayout = new QGridLayout();
        gridLayout->setObjectName("gridLayout");
        minus_button = new QPushButton(Caculator);
        minus_button->setObjectName("minus_button");
        sizePolicy.setHeightForWidth(minus_button->sizePolicy().hasHeightForWidth());
        minus_button->setSizePolicy(sizePolicy);
        minus_button->setAutoRepeat(true);
        minus_button->setAutoDefault(true);
        minus_button->setFlat(false);

        gridLayout->addWidget(minus_button, 4, 3, 1, 1);

        num1_button = new QPushButton(Caculator);
        num1_button->setObjectName("num1_button");
        sizePolicy.setHeightForWidth(num1_button->sizePolicy().hasHeightForWidth());
        num1_button->setSizePolicy(sizePolicy);
        num1_button->setAutoRepeat(true);
        num1_button->setAutoDefault(true);
        num1_button->setFlat(false);

        gridLayout->addWidget(num1_button, 5, 0, 1, 1);

        num7_button = new QPushButton(Caculator);
        num7_button->setObjectName("num7_button");
        sizePolicy.setHeightForWidth(num7_button->sizePolicy().hasHeightForWidth());
        num7_button->setSizePolicy(sizePolicy);
        num7_button->setAutoRepeat(true);
        num7_button->setAutoDefault(true);
        num7_button->setFlat(false);

        gridLayout->addWidget(num7_button, 3, 0, 1, 1);

        percent_button = new QPushButton(Caculator);
        percent_button->setObjectName("percent_button");
        sizePolicy.setHeightForWidth(percent_button->sizePolicy().hasHeightForWidth());
        percent_button->setSizePolicy(sizePolicy);
        percent_button->setAutoRepeat(true);
        percent_button->setAutoDefault(true);
        percent_button->setFlat(false);

        gridLayout->addWidget(percent_button, 6, 3, 1, 1);

        plus_button = new QPushButton(Caculator);
        plus_button->setObjectName("plus_button");
        sizePolicy.setHeightForWidth(plus_button->sizePolicy().hasHeightForWidth());
        plus_button->setSizePolicy(sizePolicy);
        plus_button->setAutoRepeat(true);
        plus_button->setAutoDefault(true);
        plus_button->setFlat(false);

        gridLayout->addWidget(plus_button, 5, 3, 1, 1);

        num5_button = new QPushButton(Caculator);
        num5_button->setObjectName("num5_button");
        sizePolicy.setHeightForWidth(num5_button->sizePolicy().hasHeightForWidth());
        num5_button->setSizePolicy(sizePolicy);
        num5_button->setAutoRepeat(true);
        num5_button->setAutoDefault(true);
        num5_button->setFlat(false);

        gridLayout->addWidget(num5_button, 4, 1, 1, 1);

        ac_button = new QPushButton(Caculator);
        ac_button->setObjectName("ac_button");
        sizePolicy.setHeightForWidth(ac_button->sizePolicy().hasHeightForWidth());
        ac_button->setSizePolicy(sizePolicy);
        ac_button->setAutoRepeat(true);
        ac_button->setAutoDefault(true);
        ac_button->setFlat(false);

        gridLayout->addWidget(ac_button, 1, 2, 1, 1);

        num6_button = new QPushButton(Caculator);
        num6_button->setObjectName("num6_button");
        sizePolicy.setHeightForWidth(num6_button->sizePolicy().hasHeightForWidth());
        num6_button->setSizePolicy(sizePolicy);
        num6_button->setAutoRepeat(true);
        num6_button->setAutoDefault(true);
        num6_button->setFlat(false);

        gridLayout->addWidget(num6_button, 4, 2, 1, 1);

        num2_button = new QPushButton(Caculator);
        num2_button->setObjectName("num2_button");
        sizePolicy.setHeightForWidth(num2_button->sizePolicy().hasHeightForWidth());
        num2_button->setSizePolicy(sizePolicy);
        num2_button->setAutoRepeat(true);
        num2_button->setAutoDefault(true);
        num2_button->setFlat(false);

        gridLayout->addWidget(num2_button, 5, 1, 1, 1);

        num3_button = new QPushButton(Caculator);
        num3_button->setObjectName("num3_button");
        sizePolicy.setHeightForWidth(num3_button->sizePolicy().hasHeightForWidth());
        num3_button->setSizePolicy(sizePolicy);
        num3_button->setAutoRepeat(true);
        num3_button->setAutoDefault(true);
        num3_button->setFlat(false);

        gridLayout->addWidget(num3_button, 5, 2, 1, 1);

        num9_button = new QPushButton(Caculator);
        num9_button->setObjectName("num9_button");
        sizePolicy.setHeightForWidth(num9_button->sizePolicy().hasHeightForWidth());
        num9_button->setSizePolicy(sizePolicy);
        num9_button->setAutoRepeat(true);
        num9_button->setAutoDefault(true);
        num9_button->setFlat(false);

        gridLayout->addWidget(num9_button, 3, 2, 1, 1);

        num4_button = new QPushButton(Caculator);
        num4_button->setObjectName("num4_button");
        sizePolicy.setHeightForWidth(num4_button->sizePolicy().hasHeightForWidth());
        num4_button->setSizePolicy(sizePolicy);
        num4_button->setAutoRepeat(true);
        num4_button->setAutoDefault(true);
        num4_button->setFlat(false);

        gridLayout->addWidget(num4_button, 4, 0, 1, 1);

        num0_button = new QPushButton(Caculator);
        num0_button->setObjectName("num0_button");
        sizePolicy.setHeightForWidth(num0_button->sizePolicy().hasHeightForWidth());
        num0_button->setSizePolicy(sizePolicy);
        num0_button->setAutoRepeat(true);
        num0_button->setAutoDefault(true);
        num0_button->setFlat(false);

        gridLayout->addWidget(num0_button, 6, 1, 1, 1);

        num8_button = new QPushButton(Caculator);
        num8_button->setObjectName("num8_button");
        sizePolicy.setHeightForWidth(num8_button->sizePolicy().hasHeightForWidth());
        num8_button->setSizePolicy(sizePolicy);
        num8_button->setAutoRepeat(true);
        num8_button->setAutoDefault(true);
        num8_button->setFlat(false);

        gridLayout->addWidget(num8_button, 3, 1, 1, 1);

        divide_button = new QPushButton(Caculator);
        divide_button->setObjectName("divide_button");
        sizePolicy.setHeightForWidth(divide_button->sizePolicy().hasHeightForWidth());
        divide_button->setSizePolicy(sizePolicy);
        divide_button->setAutoRepeat(true);
        divide_button->setAutoDefault(true);
        divide_button->setFlat(false);

        gridLayout->addWidget(divide_button, 1, 3, 1, 1);

        equal_button = new QPushButton(Caculator);
        equal_button->setObjectName("equal_button");
        sizePolicy.setHeightForWidth(equal_button->sizePolicy().hasHeightForWidth());
        equal_button->setSizePolicy(sizePolicy);
        equal_button->setAutoRepeat(true);
        equal_button->setAutoDefault(true);
        equal_button->setFlat(false);

        gridLayout->addWidget(equal_button, 6, 2, 1, 1);

        clear_history_button = new QPushButton(Caculator);
        clear_history_button->setObjectName("clear_history_button");
        sizePolicy.setHeightForWidth(clear_history_button->sizePolicy().hasHeightForWidth());
        clear_history_button->setSizePolicy(sizePolicy);
        clear_history_button->setAutoRepeat(true);
        clear_history_button->setAutoDefault(true);
        clear_history_button->setFlat(false);

        gridLayout->addWidget(clear_history_button, 1, 0, 1, 1);

        dot_button = new QPushButton(Caculator);
        dot_button->setObjectName("dot_button");
        sizePolicy.setHeightForWidth(dot_button->sizePolicy().hasHeightForWidth());
        dot_button->setSizePolicy(sizePolicy);
        dot_button->setAutoRepeat(true);
        dot_button->setAutoDefault(true);
        dot_button->setFlat(false);

        gridLayout->addWidget(dot_button, 6, 0, 1, 1);

        multiply_button = new QPushButton(Caculator);
        multiply_button->setObjectName("multiply_button");
        sizePolicy.setHeightForWidth(multiply_button->sizePolicy().hasHeightForWidth());
        multiply_button->setSizePolicy(sizePolicy);
        multiply_button->setAutoRepeat(true);
        multiply_button->setAutoDefault(true);
        multiply_button->setFlat(false);

        gridLayout->addWidget(multiply_button, 3, 3, 1, 1);

        backspace_button = new QPushButton(Caculator);
        backspace_button->setObjectName("backspace_button");
        sizePolicy.setHeightForWidth(backspace_button->sizePolicy().hasHeightForWidth());
        backspace_button->setSizePolicy(sizePolicy);
        backspace_button->setAutoRepeat(true);
        backspace_button->setAutoDefault(true);
        backspace_button->setFlat(false);

        gridLayout->addWidget(backspace_button, 1, 1, 1, 1);

        right_bracket_button = new QPushButton(Caculator);
        right_bracket_button->setObjectName("right_bracket_button");
        sizePolicy.setHeightForWidth(right_bracket_button->sizePolicy().hasHeightForWidth());
        right_bracket_button->setSizePolicy(sizePolicy);
        right_bracket_button->setAutoRepeat(true);
        right_bracket_button->setAutoDefault(true);
        right_bracket_button->setFlat(false);

        gridLayout->addWidget(right_bracket_button, 0, 3, 1, 1);

        left_bracket_button = new QPushButton(Caculator);
        left_bracket_button->setObjectName("left_bracket_button");
        sizePolicy.setHeightForWidth(left_bracket_button->sizePolicy().hasHeightForWidth());
        left_bracket_button->setSizePolicy(sizePolicy);
        left_bracket_button->setAutoRepeat(true);
        left_bracket_button->setAutoDefault(true);
        left_bracket_button->setFlat(false);

        gridLayout->addWidget(left_bracket_button, 0, 2, 1, 1);

        exit_button = new QPushButton(Caculator);
        exit_button->setObjectName("exit_button");
        sizePolicy.setHeightForWidth(exit_button->sizePolicy().hasHeightForWidth());
        exit_button->setSizePolicy(sizePolicy);
        exit_button->setAutoDefault(true);
        exit_button->setFlat(false);

        gridLayout->addWidget(exit_button, 0, 0, 1, 2);


        verticalLayout_2->addLayout(gridLayout);


        retranslateUi(Caculator);

        minus_button->setDefault(true);
        num1_button->setDefault(true);
        num7_button->setDefault(true);
        percent_button->setDefault(true);
        plus_button->setDefault(true);
        num5_button->setDefault(true);
        ac_button->setDefault(true);
        num6_button->setDefault(true);
        num2_button->setDefault(true);
        num3_button->setDefault(true);
        num9_button->setDefault(true);
        num4_button->setDefault(true);
        num0_button->setDefault(true);
        num8_button->setDefault(true);
        divide_button->setDefault(true);
        equal_button->setDefault(true);
        clear_history_button->setDefault(true);
        dot_button->setDefault(true);
        multiply_button->setDefault(true);
        backspace_button->setDefault(true);
        right_bracket_button->setDefault(true);
        left_bracket_button->setDefault(true);
        exit_button->setDefault(true);


        QMetaObject::connectSlotsByName(Caculator);
    } // setupUi

    void retranslateUi(QWidget *Caculator)
    {
        Caculator->setWindowTitle(QCoreApplication::translate("Caculator", "Caculator", nullptr));
        minus_button->setText(QCoreApplication::translate("Caculator", "-", nullptr));
        num1_button->setText(QCoreApplication::translate("Caculator", "1", nullptr));
        num7_button->setText(QCoreApplication::translate("Caculator", "7", nullptr));
        percent_button->setText(QCoreApplication::translate("Caculator", "%", nullptr));
        plus_button->setText(QCoreApplication::translate("Caculator", "+", nullptr));
        num5_button->setText(QCoreApplication::translate("Caculator", "5", nullptr));
        ac_button->setText(QCoreApplication::translate("Caculator", "AC", nullptr));
        num6_button->setText(QCoreApplication::translate("Caculator", "6", nullptr));
        num2_button->setText(QCoreApplication::translate("Caculator", "2", nullptr));
        num3_button->setText(QCoreApplication::translate("Caculator", "3", nullptr));
        num9_button->setText(QCoreApplication::translate("Caculator", "9", nullptr));
        num4_button->setText(QCoreApplication::translate("Caculator", "4", nullptr));
        num0_button->setText(QCoreApplication::translate("Caculator", "0", nullptr));
        num8_button->setText(QCoreApplication::translate("Caculator", "8", nullptr));
        divide_button->setText(QCoreApplication::translate("Caculator", "/", nullptr));
        equal_button->setText(QCoreApplication::translate("Caculator", "=", nullptr));
        clear_history_button->setText(QCoreApplication::translate("Caculator", "ClearHistory", nullptr));
        dot_button->setText(QCoreApplication::translate("Caculator", ".", nullptr));
        multiply_button->setText(QCoreApplication::translate("Caculator", "*", nullptr));
        backspace_button->setText(QCoreApplication::translate("Caculator", "BackSpace", nullptr));
        right_bracket_button->setText(QCoreApplication::translate("Caculator", ")", nullptr));
        left_bracket_button->setText(QCoreApplication::translate("Caculator", "(", nullptr));
        exit_button->setText(QCoreApplication::translate("Caculator", "exit", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Caculator: public Ui_Caculator {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CACULATOR_H
