#include <iostream>
#include <string>
using namespace std;
int main(int argc,char *argv[])
{
	//日志格式可以有设置为如下两种示例：
	//"{context} {file}:{line}"
	//"<log><context>{context}<contex> <file>{file}</file><line>{line}</line></log>"

	for (int i = 0; i < argc; i++) {
		cout << argv[i] << endl;
	}
	cout << __FILE__ << ":" << __LINE__ << endl; // 输出文件名和该代码的行号

	string fmt = "{context}-{file}:{line}"; // 设置默认日志格式

	if (argc > 2) {
		// 如果用户自行传入日志格式，则将格式改为传入的
		fmt = argv[2];
	}
	string log = "test log context 001";
	string str = fmt;

	string ckey{ "{context}" };
	auto pos = str.find(ckey);
	if (pos != string::npos)
	{
		str = str.replace(pos,ckey.size(),log);
	}

	string fkey{ "{file}" };
	pos = str.find(fkey);
	if (pos != string::npos)
	{
		str = str.replace(pos,fkey.size(),__FILE__);
	}

	string lkey{ "{line}" };
	pos = str.find(lkey);
	if (pos != string::npos)
	{
		str = str.replace(pos,lkey.size(),to_string(__LINE__));
	}

	cout << "-------------------------- log --------------------------"<<endl;
	cout << str << endl;
}
