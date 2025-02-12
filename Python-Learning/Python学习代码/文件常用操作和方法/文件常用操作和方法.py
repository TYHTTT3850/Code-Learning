# 打开文件。open(file_path,mode,encoding)
# file_path:同一文件夹下只需要文件名，否则需要绝对路径。
# mode:只读r；写入w；追加a；读写r+，w+，a+。
# encoding:编码方式。UTF-8，GBK等。
file1 = open("txt文件常用操作和方法.txt", "r", encoding="UTF-8") #file为文件对象名。
print(type(file1))
print()

#文件读取"r"。读取文件时有文件指针，每次读取会从上次读取的结尾处开始读取!!

# read方法。文件对象名.read(读取的数据长度)，不传参即为读取所有数据。
print(f"读取十个长度的内容：{file1.read(10)}")
print(f"读取全部内容：{file1.read()}")

# readlines方法。文件对象名.readlines()，按照行的方式一次性读取文件内容，返回一个列表，列表元素为每行的内容(字符串)。

# readline方法。文件对象名.readline()，一次读取一行内容，返回的数据类型为字符串。

# close方法。文件对象名.close()
file1.close()

# with open():语句，语句内操作执行完后自动关闭文件。
with open("txt文件常用操作和方法.txt","r",encoding="UTF-8") as file1:
    # for循环读取。for line in 文件对象名。一次循环读取一行内容。    
    for line in file1:
        print(line)

# 文件写入"w"。若文件不存在，则创建文件。若文件已存在，则删除原有内容在进行写入。
file2 = open("txt文件写入和追加.txt", "w", encoding="UTF-8")

# write方法。文件对象名.write(写入内容)，调用时内容并未真正写入，而是在程序内存中。
file2.write("Hello World")

# flush方法。文件对象名.flush()，刷新文件内容。
file2.flush()
file2.close() # close方法内置flush方法。

# 文件追加"a"。若文件不存在，则创建文件。若文件已存在，则在内容结尾处追加写入。
file3 = open("txt文件写入和追加.txt", "a", encoding="UTF-8")
file3.write("\nHello World")
file3.flush()
file3.close()
