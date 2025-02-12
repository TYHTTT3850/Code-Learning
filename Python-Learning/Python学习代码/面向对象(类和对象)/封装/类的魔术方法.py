# python类有许多(二三十个)内置方法，这些内置方法称为魔术方法。
# 所有形如__方法名__的成员方法都是类的内置方法，都可以叫魔术方法。
# 常用的魔术方法有__init__ __str__ __lt__ __le__ __eq__
#魔术方法也是成员方法，也需要self形参。
"""
__init__：构造方法,在设计类时使用__init__方法，可以实现在创建对象时直接传入属性的操作。
__str__：字符串方法，定义类在字符串下的表现形式
__lt__：大于，小于符号比较
__le__：大于等于，小于等于符号比较
__eq__：==符号比较
"""

class Student:
    # __init__方法：
    def __init__(self, name, age):
        self.name = name
        self.age = age
        print("创建对象成功")
     #__str__方法：
    def __str__(self):
        return f"{self.name},{self.age}"
    # __lt__方法：
    def __lt__(self, other): # other表示另一个对象
        return self.age < other.age # 这是个bool值
    # __le__方法：
    def __le__(self, other):
        return self.age <= other.age # 这是个bool值
    # __eq__方法：
    def __eq__(self, other):
        return self.age == other.age # 这是个bool值

"""__init__方法演示：创建对象时直接传入成员变量值。"""
student1 = Student("Jack",31)
student2 = Student("Mike",36)
student3 = Student("Jhon",36)

"""__str__方法演示：打印对象或将对象转换为字符串时会被调用。"""
print(student1) # 若不提供__str__方法则会输出内存地址
str1 = str(student1) # 若不提供__str__方法则str1为一串内存地址
print(str1)

""" __lt__方法演示：在对象之间进行大于和小于比较时会被调用。"""
print(student1 < student2) # 若不提供__lt__方法则会报错
print(student1 > student2)
print(student2 < student3)

"""__le__方法演示：在对象之间进行大于等于和小于等于比较时会被调用。"""
print(student1 <= student2) # 若不提供__le__方法则会报错
print(student1 >= student2)
print(student2 <= student3)

""" __eq__方法演示：在对象之间进行==符号比较时会被调用。"""
print(student1 == student2) # 若不提供__eq__方法，对象仍可以比较，但比较的是内存地址，则不同对象的比较结果一定是False
print(student2 == student3) # 提供__eq__方法则可以按照自己想要的方式进行==符号比较