#include <iostream>
#include <vector>

using namespace std;

// 普通类型求和
template <typename T>
T add(const T& a,const T& b){
    return a+b;
}

// n维数组求和
template <typename T>
vector<T> add(const vector<T> &a, const vector<T> &b){
    if(a.size() != b.size()){
        throw "The shapes of two vectors are not same";
    }
    vector<T> c(a.size());
    for(int i = 0;i < a.size(); ++i){
        c[i] = add(a[i],b[i]);
    }
    return c;
}

template <typename T>
void show(const T& elem){
    cout << elem << " ";
}

template <typename T>
void show(const vector<T>& vec){
    for(int i = 0;i<vec.size();++i){
        show(vec[i]);
    }
    cout << endl;
}

int main(){
    // 普通类型求和测试
    int a1 = 1;
    int b1 = 2;
    int c1 = add(a1,b1);
    cout << "整型求和：" << c1 <<endl;

    double a2 = 2.56;
    double b2 = 3.14;
    double c2 = add(a2,b2);
    cout << "双精度浮点型求和：" << c2 << endl;

    // 一维数组求和测试
    vector<int> vec1= {1, 2, 3};
    vector<int> vec2 = {4, 5, 6};
    vector<int> vec_sum = add(vec1, vec2);
    cout << "一维向量求和：" << endl;
    show(vec_sum);

    // 二维数组求和测试
    vector<vector<int>> vec_2d1 = {{1, 2, 3},{3,4,5}};
    vector<vector<int>> vec_2d2 = {{1, 2, 3},{3,4,5}};
    vector<vector<int>> vec_2d_sum = add(vec_2d1,vec_2d2);
    cout << "二维数组求和：" << endl;
    show(vec_2d_sum);
    return 0;
}
