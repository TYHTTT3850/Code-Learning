#include <iostream>

using namespace std;

int main() {
    //定义一个 int 类型的变量
    int x{100}; // 为x变量设置初始值，建议使用这种初始化格式，因为会这样进行"类型审查"。int x{1.3}会报错。
    //int y; // 如果不设置初始值，编译器报错(MSVC2022)
    cout << "x的值：" << x << endl;

    //通过 & 获取变量的地址
    cout << "变量x的地址" << &x << endl;

    //简单的类型强制转换： (要转成的类型)要转换的变量
    cout << "简单的强制转换:" << (long long)&x << endl;

    // 获取变量的内存大小 sizeof
    cout << "变量 x 的内存占用大小" << sizeof(x) << endl;
    long long bigint{0};
    cout <<"long long 类型内存占用:"<<sizeof(bigint)<<endl;

    return 0;
}
