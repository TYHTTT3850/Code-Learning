#include <iostream>
#include <vector>
#include <list>

using namespace std;

template <typename container>
int joseph(int n, int m){
    //初始化位置
    container knights;
    for(int i=1;i<=n;++i){
        knights.push_back(i);
    }

    typename container:: iterator it = knights.begin(); //初始化迭代器

    while(knights.size() > 1){
        int count = 1;
        for(; count<m; ++count){
            ++it;
            if (it == knights.end()){
                it = knights.begin(); //到末尾后循环至开头
            }
        }
        it = knights.erase(it); //去掉计数为m的的位置，再从下一个位置重新开始计数
        if (it == knights.end()){
            it = knights.begin(); //到末尾后循环至开头
        }
    }
    return knights.front();
}

int main(){
    cout << "vector容器实现" << endl;
    int end_num1 = joseph< vector<int> >(5,3);
    cout << end_num1 << endl;

    cout << "list容器实现" << endl;
    int end_num2 = joseph< list<int> >(5,3);
    cout << end_num2 << endl;
    return 0;
}
