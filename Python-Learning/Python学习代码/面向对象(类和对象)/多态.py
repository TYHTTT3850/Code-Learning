"""
多态：完成某个具体的行为时，使用不同的对象会得到不同的状态。
抽象类(接口)：含有抽象方法的类
抽象方法：方法体是空实现(pass)的方法
定义一个抽象类 ———— 现实生活中制定标准(顶层设计)
具体子类的实现 ———— 现实生活中实现标准(具体落实)
"""

class Animals: # 抽象类，父类决定有哪些方法，具体实现由子类决定。
    def speak(self):
        pass

class Dog(Animals):
    def speak(self):
        print("wang wang wang")

class Cat(Animals):
    def speak(self):
        print("miao miao miao")

def Make_Noise(animal:Animals): # 注解传入参数的类型为Animals类
    animal.speak()

cat = Cat()
dog = Dog()

# 结果不同
Make_Noise(cat)
Make_Noise(dog)