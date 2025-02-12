//定义一个哺乳动物类 Mammal，由此派生出狗类 Dog ，定义一个狗类对象，观察基类和派生类的构造和析构函数调用顺序

#include <iostream>
using namespace std;

class Mammal{
public:
    Mammal(){
        cout << "调用基类构造函数" << endl;
    }
    ~Mammal(){
        cout << "调用基类析构函数" << endl;
    }
};

class Dog:public Mammal{
public:
    Dog(){
        cout << "调用派生类构造函数" << endl;
    }
    ~Dog(){
        cout << "调用派生类析构函数" << endl;
    }
};

int main(){
    Dog d1; //构造函数调用顺序：基类 -> 派生类 || 析构函数：派生类 -> 基类
    return 0;
}