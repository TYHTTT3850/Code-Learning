def print_file_info(file_name):
    try:
        file = open(file_name, 'r', encoding='utf-8')
    except :
        print("文件不存在")
    else:
        print("文件存在")
        print(f"文件内容为：\n{file.read()}")
        file.close()
def append_to_file(file_name,data):
    file = open(file_name, 'a', encoding='utf-8')
    file.write(data)
    file.close()