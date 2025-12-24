#include <iostream>
#include <vector>
#include <string>
using namespace std;

//编码表
static const string base16_encode_table{"0123456789ABCDEF"};

//解码表
static const vector<int> base16_decode_table{
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,//索引0~9
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,//索引10~19
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,//索引20~29
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,//索引30~39
    -1,-1,-1,-1,-1,-1,-1,-1,//索引40~47
    0,1,2,3,4,5,6,7,8,9,//索引48~57，字符'0'~'9'
    -1,-1,-1,-1,-1,-1,-1,//58~64
    10,11,12,13,14,15,//索引65~70，字符'A'~'F'
};
int main() {
    const string tester = "测试base16的字符串";//string本质上是字节串
    string base16str;//存储base16编码后的字符串
    string decoded_result;//存储解码后的字符串

    //编码过程
    for (const unsigned char c : tester) {//取出每个字节
        //一个字节拆分为两个数字，再转换为base16编码
        const int head = c >> 4;//取二进制头4位
        const int tail = c & 0b00001111;//取二进制末4位
        base16str += base16_encode_table[head];//在编码表中找到对应字符并存入结果字符串
        base16str += base16_encode_table[tail];
    }
    cout << "原始结果：" << tester << endl;
    cout << "编码结果：" << base16str << endl;

    //解码过程
    for (int i = 0; i<base16str.size(); i+=2) {//遍历编码后的字符串
        const char head = base16str[i];//拿到高位的编码后字符
        const char tail = base16str[i+1];//拿到低位的编码后字符

        //根据字符的ASCII码通过解码表拿到字符在编码表中的位置
        //如A的ASCII码为65，编码表中A的索引为10，所以解码表中索引65存储的值为10
        const int c_head = base16_decode_table[head];
        const int c_tail = base16_decode_table[tail];

        //将高位和地位合并为一个字节
        const int c = c_head << 4 | c_tail;
        decoded_result += static_cast<char>(c);//字节存入字符串
    }
    cout << "解码结果：" << decoded_result;
    return 0;
}