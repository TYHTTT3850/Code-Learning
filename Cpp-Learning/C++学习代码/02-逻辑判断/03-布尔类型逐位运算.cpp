#include <iostream>
#include <bitset>
using namespace std;

int main() {
    bool f1{false};
    bool f2{false};
    bool t1{true};
    bool t2{true};
    cout << f1 << endl;
    cout << f2 << endl;
    cout << t1 << endl;
    cout << t2 << endl;

    // 逐位非无法运用在 bool 类型中。
    // false：0000 0000，~0000 0000 = 1111 1111
    // true：0000 0001，~0000 0001 = 1111 1110
    // bool类型做逐位非运算后不管怎么样都非0
    cout << (f1&f2)<< endl; // 注意用()包围
    cout << (f1&t1)<< endl;
    cout << (t1&t2)<< endl;
    cout << (f1|f1)<< endl;
    cout << (f1|t1)<< endl;
    cout << (t1|t2)<< endl;
    return 0;
}
