#闭包可以保存函数内的变量，而不会随着函数调用的结束而销毁
#内部函数引用外部函数的变量，这种写法就是闭包

#基础的闭包
def outer_function(num1):
    def inner_function(num2):
        return num1 + num2 #内部函数引用外部函数的变量
    return inner_function #返回内部函数对象

Closure = outer_function(10) #Closure接收内部函数对象(Closure就是inner_function)
result_1 = Closure(5) #调用内部函数
result_2 = outer_function(100)(200)#直接调用
print(result_1)
print(result_2)