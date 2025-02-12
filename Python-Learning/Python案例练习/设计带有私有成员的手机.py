"""
设计一个手机，内部有：
    1、私有成员变量：Is_5G_Enable，bool类型。
    2、私有成员方法：Check_5G()，判断 Is_5G_Enable的值。
    3、公开成员方法：Call_by_5G()。
"""
class Phone:
    __Is_5G_Enable = False

    def __Check_5G(self):
        if self.__Is_5G_Enable == True:
            print("5G开启")
        else:
            print("5G关闭，4G开启")

    def Call_by_5G(self):
        self.__Check_5G()
        print("正在通话中")

phone = Phone()
phone.Call_by_5G()
