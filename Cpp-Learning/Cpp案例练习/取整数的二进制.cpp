#include <iostream>
using namespace std;
int main()
{
    int a;
    cout << "please enter a number" <<endl;
    cin >> a;
    for (int i =0;i < 32;i++) //int类型二进制32位
    {
        cout << ((0x80000000&a)? "1":"0");
        //0x80000000对应二进制1000 0000 0000 0000 0000 0000 0000 0000
        //所以表达式0x80000000&a非0当且仅当a的最高位非0
        a = a << 1; //二进制左移一位
    }
    return 0;
}
