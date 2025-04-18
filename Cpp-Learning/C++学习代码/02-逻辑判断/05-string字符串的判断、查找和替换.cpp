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

    // 查找和替换
    string strfind{"test for find [user] test"};
    auto pos = strfind.find("[test]");
    if (pos == string::npos) {
        cout << "[test] not find" << endl;
    }
    string key = "[user]";
    pos = strfind.find(key);
    cout <<"pos == "<< pos << endl;
    auto bak = strfind;
    if (pos != string::npos) {
        cout << strfind.substr(pos) << endl;
        auto re = strfind.replace(pos,key.size(),user); //从哪个位置开始,替换几个,替换成什么
        cout << strfind << endl; // 原来的东西也被替换了,所以要再替换之前备份下来
        cout << re << endl;
        cout << bak << endl;
    }
    return 0;
}
