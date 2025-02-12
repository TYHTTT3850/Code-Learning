#include "iostream"
#include "random"
using namespace std;
int get_random_num(int min,int max)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis (min,max);
    int random_number=dis(gen);
    return random_number;
}//随机数获取函数
int main()
{
    int num = get_random_num(1,10);
    cout <<num;
    return 0;
}