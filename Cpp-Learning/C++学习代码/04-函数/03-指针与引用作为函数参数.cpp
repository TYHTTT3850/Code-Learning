#include <iostream>
using namespace std;

//指针作为参数
void ParaPtr(const int* x,const int* y) {
    cout << "x: " << *x << ", y: " << *y << endl;
    cout <<x<< ", " << y << endl;
}

//引用作为参数
void ParaRef(const int& x,const int& y) {
    cout << "x: " << x << ", y: " << y << endl;
    cout <<&x<< ", " << &y << endl;
}

void ChangeValPtr(int* x,int* y) {
    *x = 600;
    *y = 700;
}

void ChangeValRef(int& x,int& y) {
    x = 640;
    y = 740;
}

int main() {
    int x{99};
    int y{88};
    ParaPtr(&x, &y);//传递地址
    ParaRef(x, y);//传递变量
    cout << "------------------" << endl;
    ChangeValPtr(&x, &y);//指针传递会改变原变量的值
    cout << "x: " << x << ", y: " << y << endl;
    ChangeValRef(x, y);//引用传递也会改变原变量的值
    cout << "x: " << x << ", y: " << y << endl;
    return 0;

    //引用和指针作为参数可以减少复制
}
