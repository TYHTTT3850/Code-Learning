#include <iostream>
#include <string>
using namespace std;

int main() {
    while (true) {
        cout << "请输入数字";
        string line;
        int x{0};
        cin>>x;
        if (cin.fail()) {//如果输入错误
            cin.clear();//清除错误标志，恢复状态为正常
            getline(cin, line);//取出错误输入的一整行
            //如果不做上面这两步操作，cin会一直处于错误状态，无法进行后续输入
            cout << "输入错误" << endl;
            continue;
        }
        // if (cin.rdstate()==ios_base::failbit) {//如果输入错误的另一种写法
        //     cout << "输入错误，程序结束" << endl;
        //     break;
        // }
        cout << "x=" << x << endl;
        // C++ 流状态(std::ios_base::iostate)说明：
        // - goodbit : 流状态正常
        // - eofbit  : 已到达输入序列末尾
        // - failbit : 格式化输入失败
        // - badbit  : 严重底层 I/O 错误
        // 常用检测接口：
        // - good() : 仅当未置位 eofbit / failbit / badbit 时返回 true
        // - eof()  : eofbit 被置位
        // - fail() : failbit 或 badbit 被置位
        // - bad()  : badbit 被置位(最严重)
    }
    return 0;
}
