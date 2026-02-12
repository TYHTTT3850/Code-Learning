"""
用于访问 Python 解释器自身使用和维护的变量，同时模块中还提供了一部分函数，可以与解释器进行比较深度的交互。
"""

import sys

# 1、sys.argv属性：返回命令行调用 Python 脚本时提供的“命令行参数”。
print("The list of command line arguments:\n", sys.argv)

# 2、sys.platform属性：返回程序的运行平台
print("程序运行平台：",sys.platform)

# 3、sys.byteorder属性：返回运行环境的字节序，
# 字节序指的是在计算机内部存储数据时，数据的低位字节存储在存储空间中的高位还是低位。
print("字节序：",sys.byteorder)

# 4、sys.executable属性：返回当前 python 解释器的可执行文件的绝对位置
print(sys.executable)

# 5、sys.modules属性：一个字典，包含的是各种已加载的模块的模块名到模块具体位置的映射。
# print(sys.modules) # 内容过多，就不打印了

# 6、sys.builtin_module_names属性：该属性是一个字符串元组，其中的元素均为当前所使用的的 Python 解释器内置的模块名称。
# print(sys.builtin_module_names) # 内容过多，就不打印了

# 7、sys.path属性：该属性是一个由字符串组成的列表，其中各个元素表示的是 Python 搜索模块的路径；在程序启动期间被初始化。
print(sys.path)

# 8、sys.stdin：标准输入流
# 本身不是输入的内容。
# 只有当你调用它的 .read() .readline() .readlines() 等方法时，它才会从输入流中取出实际数据。
lines = sys.stdin.readlines()
for line in lines:
    print(f'行内容：{line.strip()}')

# 9、sys.stdout：标准输出流，默认指向终端(控制台)
sys.stdout.write("Hello, ")
sys.stdout.write("World!\n")

# sys.stdout可以定向到某个文件
with open('log.txt', 'w') as f:
    sys.stdout = f
    print("这段话会写入 log.txt 而不是终端")
    sys.stdout.write("这行也是\n")

# 恢复标准输出
sys.stdout = sys.__stdout__
print("恢复输出到终端")

# 10、sys.getsizeof()：与 C 语言中的sizeof运算符类似，返回的是作用对象所占用的字节数。
print(sys.getsizeof(1))