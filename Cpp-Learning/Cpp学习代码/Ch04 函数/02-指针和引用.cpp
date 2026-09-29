#include <iostream>
using namespace std;

int main() {
    int x{10};
    cout << x << endl;
    cout << &x << endl;//输出x的地址
    int* ptr=nullptr;//空指针
    int* p{&x};//指针变量p存储x的地址
    cout << *p << endl;//解析指针，输出p指向的地址存储的值
    cout << p << endl;//输出p存储的地址
    cout << &p << endl;//输出指针变量p的地址
    (*p)++;//通过指针修改x的值
    cout << x << endl;//输出x的值

    int& ref{x};//引用在定义时必须初始化
    cout << ref << endl;//输出引用ref的值
    cout << &ref << endl;//输出引用ref的地址
    ref++;//通过引用修改x的值
    cout << x << endl;//输出x的值

    //尽量使用引用，减少使用指针
    return 0;
}