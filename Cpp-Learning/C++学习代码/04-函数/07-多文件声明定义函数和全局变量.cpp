#include <iostream>
#include "Base16.h"//双引号表示以当前目录为起点查找头文件
extern "C"{
//可以引用C语言头文件
}
using namespace std;

int main() {
    GlobalVariable++;
    cout << "GlobalVariable: " << GlobalVariable << endl;
    int res = FunctionExample(5,15);
    return 0;
}