//设计一个用于人事管理的“人员”类。成员属性有：编号，性别，身份证号，出生日期。出生日期定义为内嵌子对象。
//用成员函数实现人员信息录入和显示。
//要求实现：构造函数，析构函数，复制构造函数，内联成员函数，带默认形参的成员函数，类的组合。

#include <iostream>
using namespace std;

class Date{
private:
    int year;
    int month;
    int day;
public:
    Date(int Y, int M, int D):year(Y),month(M),day(D){
        cout << "调用日期类构造函数" << endl;
    } //日期类构造函数
    Date(const Date& other){ //日期类复制构造函数
        year = other.year;
        month = other.month;
        day = other.day;
        cout << "调用日期类复制构造函数" << endl;
    }
    ~Date(){
        cout << "调用日期类析构函数" << endl;
    }
    inline void show_date(){
        cout << year << "/" << month << "/" << day << endl;
    }
};

class Person{
private:
    string num;
    string gender;
    string ID;
    Date birthday; //类的组合
public:
    Person(string N = "001", string G = "male", string I="123456", Date B= Date(1970,1,1)):num(N), gender(G), ID(I), birthday(B){
        cout << "调用人员类构造函数" << endl;
    } //带默认参数的构造函数
    ~Person(){
        cout << "调用人员类析构函数" << endl;
    }
    inline void show_person(){
        cout << "编号为" << num << "的员工的信息为：" << endl;
        cout << "性别：" << gender << endl;
        cout << "身份证号：" << ID << endl;
        cout << "出生日期：" ;
        birthday.show_date();
    }
};

int main(){
    Person p1("00001","male","332012",Date(1988,6,4));
    p1.show_person();

    return 0;
}