# include<iostream>
using namespace std;

int main() {
    //for循环基本语法：for(初始条件;终止条件;迭代方式)
    for (int i = 1; i <= 10; i++) {
        cout << i << endl;
    }

    for (int i = 1; i <= 20; i++) {
        if (i % 2 == 0) {
            continue; // continue 表示跳过这轮循环，进入到下一轮循环
        }
        cout << i << endl;
        if (i > 10) {
            break; // break 表示直接终止循环
        }
    }

    // 嵌套循环，嵌套尽量不要超过三层
    for (int i = 1; i <= 3; i++) {
        for (int j = 1; j <= 3; j++) {
            cout << i << ":" << j <<" ";
        }
        cout << endl;
    }

    // 无限循环
    // for (;;) {}
    return 0;
}