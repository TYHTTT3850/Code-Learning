//编写一个抽象类 Shape ，在此基础上派生出 Rectangle(矩形) 和 Circle(圆形) 类。
//二者都有计算面积和周长的函数

#include <iostream>
#define Pi 3.1415926
using namespace std;

class Shape{
public:
    virtual double getArea() = 0; //面积
    virtual double getPerime() = 0; //周长
};

class Rectangle:public Shape{
private:
    double length;
    double width;
public:
    Rectangle(double l, double w):length(l), width(w){}
    double getArea(){
        return length * width;
    }
    double getPerime(){
        return 2*(length + width);
    }
};

class Circle:public Shape{
private:
    double radius;
public:
    Circle(double r):radius(r){}
    double getArea(){
        return Pi * radius * radius;
    }
    double getPerime(){
        return 2 * Pi * radius;
    }
};

int main(){
    Rectangle R1(3,4);
    Circle C1(2);
    cout << "3×4矩形面积：" << R1.getArea() << endl;
    cout << "3×4矩形周长：" << R1.getPerime() << endl;
    cout << "半径为2的圆形面积：" << C1.getArea() << endl;
    cout << "半径为2的圆形周长：" << C1.getPerime() << endl;
    return 0;
}