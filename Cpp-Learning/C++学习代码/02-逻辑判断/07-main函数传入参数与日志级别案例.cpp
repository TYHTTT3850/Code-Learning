#include <iostream>
using namespace std;
enum class LogLevel
{
	DEBUG,
	INFO,
	ERROR,
	FATAL
};
int main(int argc,char *argv[]) //argc代表传入的参数个数，argv代表传入的参数数组
{
	cout << "传入执行程序时传入的参数个数：" << argc << endl;
	cout << argv[0] << endl;
	//用户传递日志的最低显示级别
	//debug < info < error < fatal
	// test_main_log info
	auto logLevel = LogLevel::DEBUG; // 默认是 DEBUG 级别的日志
	if (argc > 1) //参数大于1时才显示日志(等于1就相当于执行了exe文件，并没有传入额外的参数)
	{
		cout << argv[1] << endl;
		string levelstr = argv[1];
		if ("info" == levelstr)
			logLevel = LogLevel::INFO;
		else if("error" == levelstr)
			logLevel = LogLevel::ERROR;
		else if ("fatal" == levelstr)
			logLevel = LogLevel::FATAL;
	}

	///测试日志1 DEBUG
	{
		auto level = LogLevel::DEBUG;
		string context = "test log 1";
		if (level >= logLevel)
		{
			string levelstr = "debug";
			cout << levelstr << ":" << context << endl;
		}
	}

	///测试日志2 INFO
	{
		auto level = LogLevel::INFO;
		string context = "test log 2";
		if (level >= logLevel)
		{
			string levelstr = "info";
			cout << levelstr << ":" << context << endl;
		}
	}

	///测试日志3 ERROR
	{
		auto level = LogLevel::ERROR;
		string context = "test log 3";
		if (level >= logLevel)
		{
			string levelstr = "error";
			cout << levelstr << ":" << context << endl;
		}
	}

	///测试日志4 FATAL
	{
		auto level = LogLevel::FATAL;
		string context = "test log 4";
		if (level >= logLevel)
		{
			string levelstr = "fatal";
			cout << levelstr << ":" << context << endl;
		}
	}
	
	return 0;
}
