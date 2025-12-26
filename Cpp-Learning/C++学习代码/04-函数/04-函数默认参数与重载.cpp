#include <iostream>
using namespace std;

void DefaultPara(int x,int y = 200,int z = 300) {
    cout << "x: " << x << ", y: " << y << ", z: " << z << endl;
    // 如果函数的申明和定义分离，那么默认参数只能在声明函数时定义
}

//函数重载：函数同名，但参数类型和数量不同
void Func(int x) {
    cout << "Func with int: " << x << endl;
}
void Func(double x,double y) {
    cout << "Func with double: " << x << endl;
    cout << "Func with double: " << y << endl;
}

//函数的重载不能以返回值区分

int main() {
    Func(10);
    Func(10.5,20.5);
    return 0;
}