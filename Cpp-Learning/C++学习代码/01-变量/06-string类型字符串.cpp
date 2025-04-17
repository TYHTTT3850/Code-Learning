#include <iostream>
#include <string>
using namespace std;

int main() {
    string str{"test string 1"};
    cout << str << endl;
    str = "test string 1-2";
    string str2{str};
    cout << str2 << endl;
    string str3; // 默认空串
    cout << str3 << endl;
    return 0;
}
