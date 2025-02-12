#include "iostream"
using namespace std;
int query_money(int *money)
{
    *money = *money;
    return *money;
}
int save_money(int *money , int save)
{
    *money = *money + save;
    return *money;
}
int spend_money(int *money , int spend)
{
    *money = *money - spend;
    return *money;
}
int main ()
{
    string name ; int *money = new int;
    *money = 5000;
    bool is_run = true;
    cout <<"用户名：";
    cin >> name;
    cout <<"您好！"<< name << endl;
    while(is_run)
    {
        cout <<"您要进行的操作："<< endl;
        cout <<"查询余额：输入[1]"<< endl;
        cout <<"存入余额：输入[2]"<< endl;
        cout <<"取出余额：输入[3]"<< endl;
        cout <<"退出操作：输入[4]"<< endl;
        int num;
        cin >> num;
        switch(num)
        {
            case 1:
                query_money (money);
                cout << name <<",您现在的余额为："<< *money << endl;
                cout <<"----------查询成功----------"<< endl;
                break;
            case 2:
                int save;
                cout <<"存入金额：";
                cin >> save;
                save_money(money,save);
                cout << name <<",您现在的余额为："<< *money << endl;
                cout <<"----------存入成功----------"<< endl;
                break;
            case 3:
                int spend;
                cout <<"取出金额：";
                cin >> spend;
                if (*money < spend)
                {
                    cout <<"余额不足"<< endl;
                    cout <<"----------取出失败----------"<< endl;
                }
                else
                {
                    spend_money(money , spend);
                    cout << name <<",您现在的余额为："<< *money << endl;
                    cout <<"----------取出成功----------"<< endl;
                }
                break;
            default:
                is_run = false;
                cout <<"----------退出成功----------"<< endl;
                delete money;
                break;
        }
    }
    return 0;
}