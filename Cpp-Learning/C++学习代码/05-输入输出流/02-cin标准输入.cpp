#include <iostream>
#include <string>
using namespace std;

int main() {

    string line;
    getline(cin,line);//单行输入
    cout << line << endl;
    while (true) {
        char buffer[1024]{0};
        cout << ">>";
        cin.getline(buffer, sizeof(buffer)-1);//单行输入
        cout << "receive：" << buffer << endl;
        if (strstr(buffer, "exit")) {
            break;
        }
    }

    string cmd;
    while (true) {
        char c = cin.get();//单字符输入
        if (c=='\n') {
            cout << "cmd：" << cmd << endl;
            cmd="";
            continue;
        }
        cmd+=c;
        if (cmd=="exit") {
            break;
        }
    }

    return 0;
}