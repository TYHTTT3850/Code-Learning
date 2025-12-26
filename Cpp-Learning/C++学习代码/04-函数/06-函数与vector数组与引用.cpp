#include <iostream>
#include <vector>
using namespace std;

vector<int> VectorFunction(vector<int> vec) {
    cout << "in Function call："<< vec.data() << endl;
    return vec;//返回值优化，不会发生复制
}

vector<int> VectorRefFunction(vector<int>& vec) {
    cout << "in Function call with reference："<< vec.data() << endl;
    return vec;//返回引用，会发生复制
}

int main() {
    vector<int> vec = {1,2,3,4,5,6,7,8,9,10};
    cout << "main Function：" << vec.data()<<endl;
    vector<int> vec2 = VectorFunction(vec);//传入时会复制一份
    cout << "after function call：" << vec2.data()<< endl;
    vector<int> vec3 = VectorFunction(move(vec));//通过move传入，不会复制
    cout << "after function call with move：" << vec3.data()<< endl;
    cout << "original vec size after move：" << vec.size()<< endl;//vec被move后，就被销毁了

    vector<int> vec4 = {1,2,3,4,5,6,7,8,9,10};
    cout << "main Function" << vec4.data()<<endl;
    vector<int> vec5 = VectorRefFunction(vec4);//传入时不会复制
    cout << "after function call with reference：" << vec5.data()<< endl;
    return 0;
}