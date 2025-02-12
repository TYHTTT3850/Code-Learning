//定义一个基类，有两个函数 fn1()，fn2()，fn2()为虚函数。
//由基类派生出一个派生类，也有两个函数 fn1()，fn2()。
//在主程序中定义一个派生类对象，用基类指针和派生类指针指向派生类对象。
//分别调用 fn1()，fn2()，观察运行结果。

#include <iostream>
using namespace std;

class Base{
public:
    void fn1(){
        cout << "calling BaseClass's fn1()" << endl;
    }
    virtual void fn2(){
        cout << "calling BaseClass's fn2()" << endl;
    }
};

class Derived: public Base{
public:
    void fn1(){
        cout << "calling DerivedClass's fn1()" << endl;
    }

    void fn2(){
        cout << "calling DerivedClass's fn2()" << endl;
    }
};

int main(){
    Derived d1;
    Base* b = &d1;
    Derived* d = &d1;
    b->fn1(); //calling BaseClass's fn1()
    b->fn2(); //calling DerivedClass's fn2()
    d->fn1(); //calling DerivedClass's fn1()
    d->fn2(); //calling DerivedClass's fn2()
    return 0;
}
// 非虚函数静态绑定，虚函数动态绑定。 
