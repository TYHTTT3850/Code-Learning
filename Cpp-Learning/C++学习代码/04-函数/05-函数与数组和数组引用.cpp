#include <iostream>
using namespace std;

void ArrayFunction(int data[]) {
    cout << sizeof data << endl;
}

void ArrayFunction2(int data[],int length) {//为了数组传入函数时能获得大小，可以传入一个额外的参数表示大小
    cout << length << endl;

}

void ArrayRefFunction(int (&data)[10]) {//通过引用传入数组，可以获得数组大小
    cout << sizeof data << endl;
}

int main() {
    int data[10] = {1,2,3,4,5,6,7,8,9,10};
    cout << sizeof data << endl;
    ArrayFunction(data);//数组传入函数时会变为指针，从而无法获得大小
    ArrayFunction2(data, sizeof(data)/sizeof(data[0]));//传入数组大小
    ArrayRefFunction(data);//通过引用传入数组，可以获得数组大小
    return 0;
}