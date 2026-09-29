#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> v1;
    vector<int> v2(10);
    cout << v1.size() << endl;//存储元素数量
    cout << v2.size() << endl;
    cout << v1.capacity() << endl;//实际存储空间
    cout << v2.capacity() << endl;

    v1.resize(8);//扩充元素数量，会初始化对象
    v1.reserve(16);//扩充实际存储空间
    cout << v1.size() << endl;
    cout << v1.capacity() << endl;

    v1.clear();//清空元素，但不释放内存
    v1.shrink_to_fit();//收缩内存以适应当前元素数量
    cout << v1.size() << endl;
    cout << v1.capacity() << endl;

    // 元素空间动态变化
    for (int i = 0; i < 200; i++) {
        int last = v1.capacity();
        v1.push_back(i);
        int now = v1.capacity();
        if (now != last) {
            cout << "Capacity: " << now <<" Size："<< v1.size() << endl;
        }
    }
    return 0;
}