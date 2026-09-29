# include<iostream>
using namespace std;

int main() {
    // while循环基础语法：while(条件){}
    // 无限循环
    // while(1){} 同 while(true){}
    // while循环也同样支持 continue 和 break
    int index{0};
    bool is_exixt{false};

    while (!is_exixt) {
        cout << index << endl;
        ++index;
        if (index > 10) {
            is_exixt = true;
        }
    }
    return 0;
}
