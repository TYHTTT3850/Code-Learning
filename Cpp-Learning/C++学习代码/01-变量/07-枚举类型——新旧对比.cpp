#include <iostream>
using namespace std;

// 枚举的核心作用就是：用有意义的名字代替难以理解的数字常量，让代码表达更清晰，逻辑更明确。
// 一句话总结：数字语义化

int main() {
    // C++98 风格的传统 enum
    enum TrafficLightOld {
        RED,// 默认为0，可以手动指定，往下每个都加1，也可以手动指定
        YELLOW,
        GREEN
    };

    TrafficLightOld oldLight = GREEN;
    cout << "Old enum value: " << oldLight << endl;  // 输出 2
    int oldValue = oldLight;  // 可以隐式转换为 int
    cout << "Old as int: " << oldValue << endl;

    // C++11 的 enum class（强类型枚举）
    enum class TrafficLightNew {
        RED,//同样默认为0，也可以手动指定每个枚举项的值
        YELLOW,
        GREEN
    };

    TrafficLightNew newLight = TrafficLightNew::GREEN; //通过 :: 访问
    int newValue = static_cast<int>(newLight);  // 必须显式转换
    cout << "New as int: " << newValue << endl;  // 输出 2

    return 0;
}
