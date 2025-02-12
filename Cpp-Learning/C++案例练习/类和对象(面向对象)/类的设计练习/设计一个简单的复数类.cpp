//定义一个复数类 Complex ，实现以下功能：
//Complex c1(3,5)， 用 3+5i 初始化 c1。
//Complex c2 = 4.5， 用实数初始化c2。
//c1.add(c2)，c1与c2相加，结果存储在c1中。
//c1.show()，输出c1。

#include <iostream>
using namespace std;

class Compelex{
private:
    double _real;
    double _image;
public:
    Compelex(double real = 0.0, double image = 0.0):_real(real), _image(image){}
    void show(){
        cout << _real << "+" << _image << "i" << endl;
    }
    Compelex add(const Compelex& other){
        _real += other._real;
        _image += other._image;
        return *this;
    }
};

int main(){
    Compelex c1(3,5);
    Compelex c2 = 4.5;
    c1.add(c2);
    c1.show();
}
