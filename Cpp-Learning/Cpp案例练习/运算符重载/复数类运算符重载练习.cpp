//通过运算符重载实现复数的乘除法和指数运算
#include <iostream>
#include <cmath>
using namespace std;

class Complex{
    double real;
    double image;

public:
    double modulus = sqrt(pow(real,2) + pow(image,2));//复数的模

    Complex(double a,double b):real(a),image(b){} //构造函数

    Complex conjugate(){ //获取共轭复数
        return Complex(real,-image);
    }

    void display(){
        cout << real <<'+' << image << 'i' <<endl;
    }

    friend Complex operator* (Complex &C1,Complex &C2); //乘法重载
    friend Complex operator/ (Complex &C1,Complex &C2); //除法重载

    Complex operator^ (double n){ //幂运算重载
        const double Pi = 3.1415926;
        double theta =  atan(image/real); //获取与实轴的夹角
        return Complex(pow(modulus,n) * cos(n*theta),pow(modulus,n) * sin(n*theta));
    }
};

Complex operator*(Complex &C1,Complex &C2){
    return Complex(C1.real*C2.real-C1.image*C2.image,C1.real*C2.image+C1.image*C2.real);
}

Complex operator/ (Complex &C1,Complex &C2){
    Complex C2_ = C2.conjugate(); //取C2的共轭复数
    Complex c = C1 * C2_ ;
    c = Complex(c.real / pow(C2.modulus,2),c.image / pow(C2.modulus,2));
    return c;
}

int main(){
    Complex a = Complex(3,4);
    Complex b = Complex(5,12);

    Complex c1 = a*b;
    cout <<"c1=a*b=";
    c1.display();

    Complex c2 = b/a;
    cout <<"c2=b/a=";
    c2.display();

    Complex c3 = a^2;
    cout << "c3=a^2=";
    c3.display();
    return 0;
}
