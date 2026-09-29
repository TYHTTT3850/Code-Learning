#include <iostream>

using namespace std;

int main() {
    // bool 类型：非0都为真
    // 比较运算符：< > == != <= >=
    int x{0};
    cout << "Enter a number: ";
    cin >> x;
    // 按次序从上到下进行条件判断，当一个条件满足后，其他的条件判断就不再执行了。
    if (x > 100) {
        cout << x << " is greater than 100" << endl;

        // if语句支持嵌套，建议尽量不要超过3行
        if (x == 101) {
            cout << x << endl;
        }
        // 容易把 == 写成 =，使用下面这种写法则可以保证不出错
        if (102 == x) {
            cout << x << endl;
        }
    }
    else if (x > 200) {
        cout << x << " is greater than 200" << endl;
    }
    else {
        cout << "前面的所有条件都不满足" << endl;
    }
    return 0;
}