#include "iostream"
using namespace std;
int sum(int n)
{
    if(n == 1)
    {
        return 1;
    }
    else
    {
        return n + sum(n - 1);
    }
} //前n相求和递归
int fib(int n)
{
    if(n == 1 || n == 2)
    {
        return n - 1;
    }
    else
    {
        int res= fib(n-1) + fib(n-2);
        return res;
    }
} //斐波那契数列递归
int main ()
{
    int n;
    cin >> n;
    int res = sum(n);
    cout << res << endl;
    res = fib(n);
    cout << res << endl;
    return 0;
}
