#include <iostream>
#include <vector>

using namespace std;

int main() {
    //栈区数组
    {
        int arr1[4] {0};
        //C++11范围for循环遍历数组
        for (int a:arr1) {
            cout << a << " ";
        }
        cout << endl;
        int arr2[4];//未初始化，不确定值
        for (int a:arr2) {
            cout << a << " ";
        }
        cout << endl;
        cout << arr1[0] <<endl;//下标访问
        //数组的内存空间是连续的
        cout << &arr1[0] << " " << &arr1[1] << endl;
    }//出了大括号(作用域)，arr数组就被销毁了

    //堆区数组
    {
        int* arr = new int[4] {0};//必须分配大小
        for (int i=0;i<4;i++) {
            cout << arr[i] << " ";
        }
        cout << endl;
        delete[] arr;//堆区数组没用了之后必需要手动释放内存空间
        arr = nullptr;//释放后立即置空指针，防止野指针(指向失效的内存空间)
    }
    return 0;
}