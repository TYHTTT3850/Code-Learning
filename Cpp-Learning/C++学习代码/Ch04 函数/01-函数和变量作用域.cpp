#include <iostream>
using namespace std;

//函数定义
bool View(int index) {
    cout << "call View with index: " << index << endl;
    return true;
}

void Test(int a) {
    if (a % 2 == 0) {
        return;//可以提前终止函数运行
    }
    cout << a << endl;
}

void Setsize(int w,int h) {
    cout << &w << " " << &h << endl;
    w += 1;
    cout << w << " " << h << endl;
}

// 全局变量，进入main函数前就申请内存
int global_var = 10;

// 静态全局变量，也是进入main函数前就申请内存，但作用域仅限于本文件
static int static_global_var = 20;

void Access_Variable() {
    //静态局部变量，第一次运行此代码时才申请空间，程序结束时才释放内存
    static int static_var = 30;
    global_var++;//可以在函数内访问和修改全局变量
    static_global_var++;//也可以在函数内访问和修改静态全局变量
    static_var++;//可以在函数内访问和修改静态局部变量
}

int main() {
    bool re = View(1024);
    cout << re << endl;
    int w = 1920;
    int h = 1080;
    Setsize(w, h);//函数内的变化不会影响函数外的变量
    cout << &w << " " << &h << endl;//函数内的参数地址与函数外的变量地址不同
    cout << w << " " << h << endl;

    return 0;
}