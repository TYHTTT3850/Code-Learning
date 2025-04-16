#include <iostream>
using namespace std;

int main() {
    //逻辑运算符  (代用运算符)
    //逻辑非：!     not
    //逻辑与：&&    and
    //逻辑或：||    or
    bool f{false};
    bool t{true};
    cout << !f << endl;
    cout << not t << endl;
    cout << (f&&t) << endl;
    cout << (f and t) << endl;
    cout << (f||t) << endl;
    cout << (f or t) << endl;
    return 0;
}
