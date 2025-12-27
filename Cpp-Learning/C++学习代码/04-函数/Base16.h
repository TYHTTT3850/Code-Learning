//保证单个cpp中只会引用一次，支持MSVC GCC CLANG
#pragma once//编译效率更高
#ifndef BASE16_H//兼容性更好
#define BASE16_H
//声明全局变量
extern int GlobalVariable;
//一般不要在头文件中定义全局变量
//int x;//错误示范，可能引起重复定义错误

//函数声明
int FunctionExample(int a, int b=10);//函数声明和定义分离式，参数默认值必须在声明时指定

//C语言函数
extern "C" void C_FunctionExample();
extern "C"{//声明一组C语言函数
    void C_FunctionExample2();
    void C_FunctionExample3();
    void C_FunctionExample4();
}
#endif