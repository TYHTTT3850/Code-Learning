#include "Base16.h"
int GlobalVariable;

//有声明默认参数，则定义时不能定义默认值
int FunctionExample(int a, int b) {
    return a+b;
}

//定义C语言函数
extern "C" void C_FunctionExample() {
    //C语言函数实现
    //可以使用C++特性
    //函数对外暴露为C函数而不是C++函数
    //通常用于多语言联合编程，为其他语言提供接口
}

//定义一组C语言函数
extern "C"{
    void C_FunctionExample2() {

    }
    void C_FunctionExample3() {

    }
    void C_FunctionExample4() {
    }
}
