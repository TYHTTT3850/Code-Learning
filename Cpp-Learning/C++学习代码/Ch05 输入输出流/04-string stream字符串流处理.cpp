#include <iostream>
#include <string>
#include <sstream>//string stream
using namespace std;

int main() {
    //string stream基础用法
    stringstream ss1;
    ss1 << "test string stream"<<100;//和cout用法类似
    cout<<ss1.str()<<endl;//输出字符串流内容
    ss1<<boolalpha;//直接输出bool值true/false
    ss1<<hex;//十六进制输出
    ss1<<"\n"<<false<<100<<endl;
    cout<<ss1.str()<<endl;
    ss1<<noboolalpha;//恢复成数字0/1输出
    ss1<<dec;//恢复成十进制输出
    ss1.str("");//清空字符串流内容
    cout<<ss1.str()<<endl;

    //string stream格式输入
    string data1 = "test1 test2 test3";
    stringstream ss2(data1);
    string temp;
    // 使用 string stream 将字符串按“空白字符”进行分词。
    // 运算符 >> 在读取 string 时会自动跳过前导空白，并读取到下一个空白字符为止，因此每次 >> temp都会依次得到 "test1"、"test2"、"test3"。
    ss2>>temp;cout<<temp<<",";
    ss2>>temp;cout<<temp<<",";
    ss2>>temp;cout<<temp<<",";
    cout<<endl;

    //string stream 单行读取
    string data2 = "test1 test2 test3\ntest4 test5 test6\ntest7 test8 test9";
    stringstream ss3;
    ss3.str(data2);
    string line;
    while (true) {
        getline(ss3, line);//从字符串流ss3中读取一行到line中
        cout<<line<<endl;
        if (ss3.eof()) break;//若到达流末尾则退出循环
    }
    return 0;
}