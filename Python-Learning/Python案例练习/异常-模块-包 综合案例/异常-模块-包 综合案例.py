from my_packages import str_tools
from my_packages import file_tools
str = "hello world"
str_tools.str_reverse(str)
str_tools.substr(str,0,4)
file_tools.print_file_info("异常-模块-包 综合案例.txt")
file_tools.append_to_file("异常-模块-包 综合案例.txt","\n你好 世界")
file_tools.print_file_info("异常-模块-包 综合案例.txt")