"""
类中除了公开的(可以在类外部调用)的成员变量和成员方法外，还有私有的(只能在类内部调用)成员变量和方法
私有的成员变量和方法对应现实世界中不对普通用户公开的属性和行为。如手机运行电压和驱动信息(属性)，程序调度和内存管理(方法)
为了使用这些私有的属性和行为，需要使用特殊手段突破权限，如苹果手机越狱，安卓手机ROOT
"""
class Phone:
    IMEI = None
    producer = None

    # 定义私有成员变量。__私有变量名
    __current_voltage = 0.9

    # 定义私有成员方法。def __私有方法名:
    def __Keep_Single_core(self):
        print("单核模式运行")

    # 在类内部调用私有成员和方法
    def Call_by_5G(self):
        if self.__current_voltage >= 1:
            print("启用5G通话")
        else:
            self.__Keep_Single_core()
            print("关闭5G通话")

phone = Phone()

try: #尝试在类外部(创建的对象)调用私有成员变量
    print(phone.__current_voltage)
except Exception as e:
    print(e) #报错

try: #尝试在类外部(创建的类对象)调用私有成员方法
    phone.__Keep_Single_core()
except Exception as e:
    print(e) #报错

phone.Call_by_5G() # 不报错