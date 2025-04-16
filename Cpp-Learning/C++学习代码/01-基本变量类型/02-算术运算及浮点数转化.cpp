#include <iostream>

using namespace std;

int main() {
    // 字面量
    cout <<"int 类型字面量：" << 123 << "," << -123 << endl;
    cout <<"long long 类型字面量：" << 123LL << "," << " sizeof(123LL)：" << sizeof(123LL) << endl;
    cout <<"默认的浮点数字面量为double类型："<< "sizeof(3.3)：" <<sizeof(3.3) << endl;
    cout << "指定浮点数字面量为float型：" << "sizeof(2.3f)：" << sizeof(2.3f) << endl;

    // 基本的算术运算,加(+)、减(-)、乘(*)、除(/)、取余(%)，均支持字面量和变量
    int x{1};
    int y = x+1;
    x += 1; // x = x + 1，以下同理
    x -= 1;
    x *= 2;
    x /= 2;
    x %= 2;
    x++; // 更简单的自增1
    x--;
    ++x; // 推荐使用这种写法，因为减少了一个临时变量
    --x;

    // 除法的特殊性(不能除以0)
    cout << "4/3 = " << 4/3 << endl; //结果为0(两个整数相除会丢弃小数位)
    cout << "1/2. = " << 1/2. << endl; //保证其中一个为浮点数

    // 浮点数转换为整数不会四舍五入，而是直接丢弃小数位
    int u;
    u = 1.3;
    cout << u << endl;
    return 0;
}
