#include <iostream>
#include <bitset>
using namespace std;

int main() {
    // 逐位运算符：用于二进制运算，对二进制数的每一位进行比较
    // 逐位非：~，~0=1 ~1=0
    // 逐位与：&，1&1=1 1&0=0 0&0 = 0
    // 逐位或：|，1|1=1 1|0=1 0|0=0
    char a = 0b10000001; // 0b开头表示二进制，c++14及以后支持
    char b = 0b10000000;
    cout << "a:" << bitset<8>(a) <<endl;
    cout << "b:" << bitset<8>(b) <<endl;
    cout << "~a:" << bitset<8>(~a) <<endl;// 逐位非
    cout << "a&b:" << bitset<8>(a&b) <<endl;// 逐位与
    cout << "a|b:" << bitset<8>(a|b) <<endl;// 逐位或
    return 0;
}
