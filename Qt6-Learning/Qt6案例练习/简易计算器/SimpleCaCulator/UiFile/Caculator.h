//
// Created by Tyh11 on 25-1-21.
//

#ifndef CACULATOR_H
#define CACULATOR_H

#include <QWidget>


QT_BEGIN_NAMESPACE
namespace Ui { class Caculator; }
QT_END_NAMESPACE

class Caculator : public QWidget {
Q_OBJECT

public:
    explicit Caculator(QWidget *parent = nullptr);
    ~Caculator() override;

private slots:
    void enterIntoLineEdit();
    void backspaceIntoLineEdit();
    void caculateResult();
    void percentageIntoLineEdit();

private:
    Ui::Caculator *ui;
};


#endif //CACULATOR_H
