#include <iostream>
#include <string>
using namespace std;

int main() {
    // 字符串长度
    string str4{"123456789"};
    cout << "str4存储了几个字符，不包括\\0：" << str4.size() << endl;
    cout << "str4实际能用的空间" << str4.capacity() << endl;

    // 截断字符串
    cout << str4.substr(3) << endl; //从索引为 3 处开始取到最后
    cout << str4.substr(1, 3) << endl; // 从索引为 1 处开始,取 3 个

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
