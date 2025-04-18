#include <iostream>
#include <string>
using namespace std;

int main() {
    // 空串判断
    string strif;
    if (strif.empty()) {// 最高效,一般都用这个
        cout << "strif is empty" << endl;
    }
    if (strif.size() == 0) {
        cout << "strif.size() == 0" << endl;
    }
    if (strif == "") {
        cout << "strif == " << endl;
    }

    // string也可以像普通变量那样做逻辑判断
    string strif2{"test"};
    cout << (strif == strif2) << endl;
    return 0;
}
