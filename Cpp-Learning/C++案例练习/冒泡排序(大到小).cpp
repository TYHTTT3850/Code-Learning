#include "iostream"
using namespace std;
int main()
{
    int *p = new int[10] {3,5,1,11,99,66,22,2,8,6};
    //每走一趟总会有一个数到达指定位置，所以只要走N-1趟
    for(int i=0;i<9;i++) //第i趟“冒泡”
    {
        for(int j=0;j<9-i;j++) //第i趟已有i个元素到达指定位置，则此i个已经排好位置的元素就不需要参与比较了。j+1<10-i
        {
            if(p[j]<p[j+1]) //从小到大则只需从“<”变为“>”。
            {
                int temp; //定义一个临时变量
                temp = p[j+1];
                p[j+1]=p[j];
                p[j]=temp;
            }
        }
    }
    for(int i =0;i<10;i++)
    {
        cout <<p[i]<<" ";
    }
    cout <<endl;
    return 0;
}