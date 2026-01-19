#include <iostream>
#include <string>
#include <fstream>//既支持读又支持写
//模式说明：
//out——写入文件 binary——以二进制方式读写文件(\r \n 不处理) app——追加写入文件 in——读取文件
using namespace std;

int main() {
    const string filename="test_file.txt";

    //fstream
    fstream write_file_1(filename, ios::out|ios::binary); // 打开文件用于写入，默认会清空原内容
    //另一种打开方式
    // write_file.open("test_file.txt", ios::out|ios::binary);
    write_file_1<<"123456789\n";//类似于cout的用法，向文件中写入。
    write_file_1.close();

    //ofstream
    ofstream write_file_2(filename, ios::app|ios::binary);//追加写入
    write_file_2<<"abcdefg\n"<<flush;
    write_file_2.write("1234",4);//以二进制方式写入
    write_file_2.close();
    return 0;
}
