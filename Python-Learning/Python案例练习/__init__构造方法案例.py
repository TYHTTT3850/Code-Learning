# 设计一个类，使用__init__构造方法录入学生信息(名字，年龄，地址)并输出。
class Student:
    def __init__(self, name, age, address):
        self.name = name
        self.age = age
        self.address = address
        print(f"录入成功，学生信息：【姓名：{self.name}，年龄：{self.age}，地址：{self.address}】"
              ,end="\n\n")

list_of_objects = [] #创建一个对象列表

for i in range(2):
    print(f"第{i+1}位同学信息")
    name = input("输入学生姓名：")
    age = int(input("输入学生年龄"))
    address = input("输入学生地址")
    list_of_objects.append(Student(name,age,address)) #将创建的对象存入列表中

for obj in list_of_objects:
    print(obj.name, obj.age, obj.address, sep=" ")