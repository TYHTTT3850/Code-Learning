//定义一个 Point 类，有数据成员 x，y。定义友元函数实现重载运算符 "+" 。
//重载 "<<" 运算符实现对象的输出。
#include <iostream>
using namespace std;

class Point{
    double _x;
    double _y;
public:
    Point(double x, double y):_x(x),_y(y){}
    friend Point operator +(Point P1, Point P2);
    friend ostream& operator<< (ostream& os, const Point& P);
    
};

Point operator +(Point P1, Point P2){
    return Point(P1._x + P2._x, P1._y + P2._y);
}

ostream& operator<< (ostream& os, const Point& P){
    os << "(" << P._x << "," << P._y << ")";
    return os;
}

int main(){
    Point P1(2,2);
    Point P2(3,4);
    Point P3 = P1 + P2;
    cout << P3 << endl;
    return 0;
}