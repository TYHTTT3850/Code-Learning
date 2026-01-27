#nonlocal关键字：能够让内部函数修改外部函数的变量值

def outer_function():
    a = 100 #外部函数变量
    def inner_function():
        # a = a + 1 #报错，因为内部函数无法修改外部函数变量
        nonlocal a #内部函数修改外部函数的变量值
        a = a + 1
        print(a)
    return inner_function

Closure = outer_function()
Closure() #101
Closure() #102
Closure() #103