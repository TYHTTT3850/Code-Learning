"""
继承:1.基于旧的类进行修改得到新的类。分为单继承和多继承
    2.被继承的类叫做父类
    3.继承后得到的新类叫做子类
    4.如果多继承的父类中有相同的成员变量(方法)，该成员变量(方法)的内容取决于继承时位置最靠前的父类。
      也就是先继承的父类优先级高于后继承的父类
复写:子类继承父类后，在子类中重新定义与父类同名的成员变量和方法
"""
class Phone:
    IMEI = "123456"
    producer = "A"
    def call_by_4g(self):
        print("4g通话")

class Phone1(Phone):
    face_id = 1001
    def call_by_5g(self):
        print("新功能5g通话")

# 单继承，子类继承一个父类。class 类名(父类名):
phone_1 = Phone1()
print(phone_1.IMEI,phone_1.producer,phone_1.face_id,sep=",") # 父类属性被继承
phone_1.call_by_4g() # 父类方法被继承
phone_1.call_by_5g()
print()

# 多继承，子类继承多个父类。class 类名(父类名1,父类名2,...):
class NFC:
    NFC_type = "5gen"
    NFC_producer = "B"
    def Write(self):
        print("NFC写入模式")
    def Read(self):
        print("NFC读取模式")

class RemoteControl:
    rc_type = "红外遥控"
    def turn_on(self):
        print("红外遥控开启")

class Phone2(Phone,NFC,RemoteControl):
    pass # 表示不再需要加入其他成员变量和方法了。关键字pass即为空，和什么都不写是一样的，写pass是为了语法不报错。

phone_2 = Phone2()
print(phone_2.IMEI,phone_2.producer,phone_2.NFC_type,phone_2.NFC_producer,phone_2.rc_type,sep=" ")
phone_2.call_by_4g()
phone_2.Read()
phone_2.Write()
phone_2.turn_on()
print()

# 复写
class Phone3(Phone):
    # 复写成员变量
    IMEI = "789012"
    producer = "B"
    # 复写成员方法
    def call_by_4g(self):
        print("开启CPU单核模式")
        print("4g通话")
phone_3 = Phone3()
print(phone_3.IMEI,phone_3.producer,sep=" ")
phone_3.call_by_4g()
print()

# 调用父类成员变量和方法。
#1. 父类名.成员变量 | 父类名.成员方法(self)
class Phone4(Phone):
    def call_by_4g(self):
        print("开启CPU能效模式")
        print(Phone.IMEI)
        Phone.call_by_4g(self)
phone_4 = Phone4()
phone_4.call_by_4g()
print()

#2. super().成员变量 | super().成员方法()
class Phone5(Phone):
    def call_by_4g(self):
        print("开启CPU效率模式")
        print(super().IMEI)
        super().call_by_4g()
phone_5 = Phone5()
phone_5.call_by_4g()