#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    //vector基础用法，<>中可以加入任何类型，甚至可以嵌套好几层
    vector<int> vi(10);//初始化10个元素，默认值为0
    vector<float> vf{1.1,2.2,3.3};//初始化3个元素
    vector<string> vs{"s1","s2","s3"};

    vi.push_back(100);//尾部增加数据

    //下标访问
    for (int i = 0; i < vs.size(); i++) {
        cout << vs[i] << " ";
    }
    cout << endl;

    //迭代器访问
    for (auto it = vf.begin(); it != vf.end(); it++) {
        cout << *it << " ";//和普通指针一样用*取值
    }
    cout << endl;

    //范围for循环访问
    for (auto val : vi) {
        cout << val << " ";
    }
    cout << endl;

    //find函数查找元素
    auto result = find(vs.begin(), vs.end(), "s2");
    if (result != vs.end()) {
        cout << "Found: " << *result << " in " << result-vs.begin() << endl;
    } else {
        cout << "Not Found" << endl;
    }

    //查找多个元素
    vector<int> vnum{1,2,3,4,5,3,6,3};
    int target = 3;
    vector<int> positions;
    for (int i = 0; i < vnum.size(); i++) {
        if (vnum[i] == target) {
            positions.push_back(i);
        }
    }

    //从前往后删除会对后面的元素位置产生影响，所以从后往前删除
    for (int i = positions.size() - 1; i >= 0; --i) {
        vnum.erase(vnum.begin() + positions[i]);
    }
    for (auto it = vnum.begin(); it != vnum.end(); it++) {
        cout << *it << " ";
    }
    cout << endl;

    //插入元素
    auto f = find(vs.begin(), vs.end(), "s1");
    vs.insert(f,"s1_1");
    for (auto val : vs) {
        cout << val << " ";
    }
    cout << endl;

    vector<int> vsort{5,2,8,1,4};
    //正序排序
    sort(vsort.begin(), vsort.end());
    for (auto val : vsort) {
        cout << val << " ";
    }
    cout << endl;
    //倒序排序
    sort(vsort.begin(), vsort.end(), greater<int>());
    for (auto val : vsort) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}