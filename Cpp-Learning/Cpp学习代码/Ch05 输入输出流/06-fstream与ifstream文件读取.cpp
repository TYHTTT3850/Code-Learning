#include <iostream>
#include <string>
#include <fstream>//既支持读又支持写
using namespace std;

int main() {
    const string filename="test_file.txt";

    //fstream
    fstream read_file_1(filename, ios::in|ios::binary);
    if (!read_file_1.is_open()) {
        cerr << "Error opening file for reading: " << filename << endl;
        return -1;
    }
    char buffer[1024]{0};
    read_file_1.read(buffer, sizeof(buffer)-1);//减1是为了给字符串结尾留出\0
    cout<<read_file_1.gcount()<<endl;//gcount()返回上次读取的字节数
    cout<<buffer<<endl;
    read_file_1.close();

    //ifstream
    //获取文件大小，ios::ate表示文件指针移到文件末尾
    ifstream read_file_2(filename, ios::ate|ios::binary);
    cout<<read_file_2.tellg()<<endl;//tellg()读取当前位置的指针位置
    //读取文件追加的内容
    string line;
    while (true) {
        getline(read_file_2, line);
        if (!line.empty()) cout<<"line："<<line<<endl;
        if (line=="exit") break;
        read_file_2.clear();//清除eof标志,文件流就可以继续使用
    }
    return 0;
}