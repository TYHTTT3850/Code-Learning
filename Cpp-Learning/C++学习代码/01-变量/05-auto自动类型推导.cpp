#include <iostream>

using namespace std;

int main() {
    // auto 自动类型推导。当代码的类型比较复杂时，会大大简化代码的复杂度(泛型编程)
    // 当变量的类型变化时，代码可以不变
    // 建议优先使用
    auto a1{10}; // 用auto a1 = 10;这种格式也一样，两种格式对 auto 来说效果一样。 
    auto a2{20.};
    // 使用 typeid(数据).name()查看数据的类型
    cout << "a1类型：" << typeid(a1).name() << endl;
    cout << "a2类型：" << typeid(a2).name() << endl;

    // 注意：当将常量赋值给 auto 类型时，会自动推到为普通变量。
    constexpr int cx{1};
    auto a3{cx};
    cout <<"a3类型：" << typeid(a3).name() << endl;

    //constexpr auto a = a1 + a2; //涉及到运算，编译失败

    const auto a4{cx}; // 要使推导的类型也为常量，手动加上 const
    // a4 = 10; //报错
    cout <<"a4类型：" << typeid(a4).name() << endl;// 常量并不是新的类型，只不过是原来的数据类型加上修饰符 const 变成只读的而已，所以输出的仍然是int。
    return 0;
}
