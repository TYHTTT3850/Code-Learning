#include "iostream"
using namespace std;
int main()
{
    int n;
    cout <<"倒金字塔层数"<< endl;
    cin >> n;
    for(int i=n;i>=1;i--)
    {
        for(int j = n-i;j>=1;j--)
        {
            cout <<" ";
        }
        for(int j = 2*i-1;j>=1;j--)
        {
            cout <<"*";
        }
        cout << endl;
    }
    return 0;
}