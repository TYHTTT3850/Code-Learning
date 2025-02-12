# 将文本操作综合案例.txt中的数据进行备份，其中类型为测试的数据剔除。
file = open("文本操作综合案例.txt","r",encoding="utf-8")
file_backup = open("备份.txt","w",encoding="utf-8")
for line in file:
    if line.count("测试") == 1:
        continue
    elif line.count("测试") == 0:
        file_backup.write(line)
file.close()
file_backup.close()