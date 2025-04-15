#include <iostream>

using namespace std; // 使用std命名空间

// 注释

int main() {
    //定义一个 int 类型的变量
    int x{100}; // 为x变量设置初始值
    //int y; // 如果不设置初始值，编译器报错(MSVC2022)
    cout << "x的值：" << x << endl;

    //通过 & 获取变量的地址
    cout << "变量x的地址" << &x << endl;

    //简单的类型强制转换 (类型)
    cout << "简单的强制转换:" << (long long)&x << endl;

    // 获取变量的内存大小 sizeof
    cout << "变量 x 的内存占用大小" << sizeof(x) << endl;
    long long bigint{0};
    cout <<"long long 类型内存占用:"<<sizeof(bigint)<<endl;

    return 0;
}
