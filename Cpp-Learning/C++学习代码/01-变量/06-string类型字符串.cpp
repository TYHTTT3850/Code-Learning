#include <iostream>
#include <string>
using namespace std;

int main() {
    // 定义与初始化
    string str{"test string 1"};
    cout << str << endl;
    str = "test string 1-2";
    string str2{str};
    cout << str2 << endl;
    string str3; // 默认空串
    cout << str3 << endl;

    // 字符串长度
    string str4{"123456789"};
    cout << "str4存储了几个字符，不包括\\0：" << str4.size() << endl;
    cout << "str4实际能用的空间" << str4.capacity() << endl;

    // 截断字符串
    cout << str4.substr(3) << endl; //从索引为 3 处开始取到最后
    cout << str4.substr(1, 3) << endl; // 从索引为 1 处开始,取 3 个

    // 字符串转换为数字
    auto i1 = stoi("1234"); // 转换为整数
    cout << i1 << endl;
    auto d1 = stod("1234.5"); //转换为浮点数
    cout << d1 << endl;
    auto ll1 = stoll("12333452241431"); //转换为long long 类型
    cout << ll1 << endl;

    // 数字转换为字符串
    auto pistr = to_string(3.1415926);
    cout << pistr << endl;

    // 字符串拼接
    string log;
    string txt{"login success"};
    string user{"admin"};
    int thread_id{123};
    log = user +":"+ txt + to_string(thread_id);
    log = "[debug]" + log;
    cout << log << endl;
    return 0;
}
