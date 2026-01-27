"""
os 模块主要用于与操作系统交互
"""

import os

"""----------文件和目录操作----------"""
# 获取当前工作路径
print(os.getcwd())

# 改变当前工作目录
#os.chdir('/path/to/directory')

# 列出指定目录的文件和子目录
print(os.listdir("../.."))

# 检测文件(目录)是否存在
print("是否存在:", os.path.exists("01-os模块常用方法.py"))

# 创建单层目录
if os.path.exists("new directory"):
    print("目录已存在")
else:
    print("目录不存在，创建")
    os.mkdir("new directory")

# 递归地创建多层目录
if os.path.exists("new/directory"):
    print("目录已存在")
else:
    print("目录不存在，创建")
    os.makedirs("new/directory")

# 删除单层目录
if os.path.exists("new directory"):
    print("目录存在，删除")
    os.rmdir("new directory")
else:
    print("目录不存在")

# 递归地删除多层目录
if os.path.exists("new/directory"):
    print("目录存在，删除")
    os.removedirs("new/directory")
else:
    print("目录不存在")

# 重命名文件或目录
if os.path.exists("test.txt"):
    os.rename("test.txt", "new.txt")
else:
    pass

# 删除文件
if os.path.exists("test.txt"):
    os.remove("test.txt")
else:
    pass

# 获取文件信息
print(os.stat("01-os模块常用方法.py"))

# 获取文件大小,以字节为单位
print(os.path.getsize("01-os模块常用方法.py"))

"""----------路径操作----------"""
# 路径操作主要在 os.path 子模块中

# 连接路径,会自动处理操作系统路径分隔符问题
print(os.path.join("D:"+os.sep,"test0","test1","test.txt")) #os.sep表示当前操作系统路径分隔符

# 获取文件(目录)在当前系统中的绝对路径
print(os.path.abspath('01-os模块常用方法.py'))

# 分割目录和文件
print(os.path.split("D:/test0/test1/test.txt")) #分割为目录+文件

# 判断是否为文件
print(os.path.isfile('new.txt'))

# 判断是否为目录
print(os.path.isdir('../01-os模块'))

"""----------环境变量----------"""

# 获取操作系统中设置的环境变量
print(os.environ)

# 获取特定环境变量的值
print(os.environ.get("path"))

"""----------进程管理----------"""

# 执行程序或命令command,在Windows系统中,返回值为cmd的调用返回信息
os.system('C:\\Windows\\System32\\calc.exe')

# 获取当前进程id
print(os.getpid())

"""----------系统信息----------"""

# 获取系统名称
print(os.name) # 返回 'posix'(Linux/Mac), 'nt'(Windows), 'java'

# 获取cpu核心数
print(os.cpu_count())

# 获取当前登录用户名称
print(os.getlogin())
