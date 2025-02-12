"""
类型注解: 在代码涉及交互的地方提供数据类型的注解。
     帮助IDE进行代码提示，帮助开发者对变量进行类型注释。
     类型注解常在无法直接看出变量类型的时候使用。
     不强制规定变量必须为某个类型，只是一种提示。即仅是提示性的而非决定性的。
"""
# 变量:类型。
# 基础数据类型注解。
var1:int = 1
var2:float = 3.1415926
var3:bool = True
var4:int = 1.25 # 不强制规定变量必须为某个类型
print(var1, var2, var3, var4,sep=" ")

# 数据容器注解。
# 简单注解。变量:类型
list1:list = [1,2,3]
tuple1:tuple = (1,2,3)
set1:set = {1,2,3}
dict1:dict = {'a':1,'b':2,'c':3}
str1:str = "hello"
# 详细注解。变量:类型[内部元素类型]
list2:list[int] = [4,5,6] # 表示列表中的元素全是int
tuple2:tuple[str,int,bool] = ("a",5,True) # tuple详细注解需要把每个元素的类型都标记出来
set2:set[int] = {4,5,6} # 表示集合中的元素全是int
dict2:dict[str,int] = {'a':4,'b':5,'c':6} # str表示key为字符串，int表示value为整数

# 类对象注解。类对象:类名
class Student:
    ID = 226002262
student1:Student = Student()
print(student1.ID)

# 在注释中标记变量的数据类型。#type:类型。对容器注解时同样可以进行详细注解。
var5 = 2 #type:int
var6 = 2.5 #type:float

# 类型注解常在无法直接看出变量类型的时候使用