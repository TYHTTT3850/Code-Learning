"""
封装是面向对象编程的一个特点。指的是将现实世界中事物的属性和行为描述为程序世界中类的成员变量和成员方法
类(class)： 1.类的属性(定义在类中的变量,称为成员变量)
           2.类的行为(定义在类中的函数,称为成员方法)
           把定义在类内部的函数称作方法
类和对象的关系：类就像是“生产图纸 ”，设计类的时候就像在“设计生产图纸 ”。
             对象就像“产品”，用类创建对象时就像根据“设计图纸(类)”生产“产品(对象)”
"""

# 定义类。class 类名称：成员变量 成员方法
class student:
    # 定义成员变量并赋予默认值None(空或缺失)
    name = None
    age = None
    # 定义方法。def 方法名(self,形参1,...,形参N)
    # 在类中定义方法时，必须含有self形参，self表示类对象自身。
    # 调用方法时，self会被自动传入，无需传参且透明(不占用参数位置)，不用理会，当其不存在。
    # 在方法内部访问成员变量必须使用self。self.类中的变量名。
    def SayHello1(self):
        print(f"Hello I'm {self.name}.I'm {self.age} years old.")
    def SayHello2(self,msg):print(msg)

# 创建类对象。对象名 = 类名称()
stu1 = student()
stu2 = student()

# 修改成员变量的值
stu1.name = "Mike"
stu2.name = "Marry"
stu1.age = 13
stu2.age = 14

# 调用成员变量。对象名.属性名
print(stu1.name,stu2.name,sep=" , ")
print(stu1.age,stu2.age,sep=" , ")

# 调用成员方法。对象名.方法名(实参)
stu1.SayHello1()
stu2.SayHello1()
stu1.SayHello2("aaaaaa") #调用方法时无需理会self
stu1.SayHello2("bbbbbb") #调用方法时无需理会self
"""
区别成员变量和成员方法：调用时是否有(),成员变量没有，成员方法有
"""