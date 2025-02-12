#include "iostream"
using namespace std;
int main()
{
    int *p1 = new int[5] {1,3,5,7,9};
    int *p2 = new int[3];
    int *p3 = new int[7];
    int offset;
    //数组删除元素
    offset =0;
    for (int i =0;i<5;i++)
    {
        if (i==1)
        {
            offset++;
            continue;
        }
        else if (i==3)
        {
            offset++;
            continue;
        }
        else
        {
            p2[i-offset]=p1[i];
        }
    }
    for (int i=0;i<3;i++)
    {
        cout <<p2[i]<<" ";
    }
    cout <<endl;
    //数组增加元素
    offset =0;
    for (int i=0;i<5;i++)
    {
        if (i==1)
        {
            p3[i+offset]=11;
            offset++;
            p3[i+offset]=p1[i];
        }
        else if(i==3)
        {
            p3[i+offset]=22;
            offset++;
            p3[i+offset]=p1[i];
        }
        else
        {
            p3[i+offset]=p1[i];
        }
    }
    for (int i=0;i<7;i++)
    {
        cout <<p3[i]<<" ";
    }
    cout <<endl;
    delete[] p1;
    delete[] p2;
    delete[] p3;
    return 0;
}