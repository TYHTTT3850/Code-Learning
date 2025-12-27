#include <iostream>
using namespace std;


int main() {
    //cout 标准输出格式
    cout<<"test cout"<<endl;
    cout<<100<<endl;//默认十进制输出
    cout<<oct<<100<<endl;//八进制输出
    cout<<100<<endl;//仍然是八进制输出
    cout<<hex<<100<<endl;//十六进制输出
    cout<<dec<<100<<endl;//恢复成十进制输出
    cout<<true<<":"<<false<<endl;
    cout<<boolalpha<<true<<endl;//以true和false输出布尔值

    //cout无格式输出
    cout.put('A').put('B');//单个内容
    cout.put('C');
    cout.put(68);//ASCII码68对应字符'D'
    cout.write("123",3);//多个内容
    string str="hello world";
    cout.write(str.c_str(),str.size());

    cout<<flush;//刷新输出缓冲区
    return 0;
}
