#include <iostream>

using namespace std;

int main() {
    // 常量：不允许变化的量

    // 运行时常量 const。涉及到传递或计算得到结果时，想要使结果不可更改时，用这个。
    int a{30};
    int b{70};
    const int cx{a + b};
    cout <<"运行时常量"<<cx<<endl;
    // cx = 200; //报错

    // 编译时常量 constexpr。申请一个不能更改的值时，使用这个。
    constexpr int expr{200};
    // constexpr int sum{a*10}; // 错误，必须在编译时就能确定，a*10只有在运行时能确定
    cout <<"编译时常量"<<expr<<endl;
    return 0;
}
