# 多返回值函数
def multiple_returns():
    return 1,"hello",True # 可以任意多
x,y,z = multiple_returns() #位置对应
print(x,y,z,sep=",")
print()

# 位置参数和关键字参数
def user_info1(name,age,gender):
    print(f"Hello {name},you are {age},you are {gender}")
user_info1("John",23,"Male") #位置参数
user_info1(name="Marry",age=23,gender="Female") #关键字参数
user_info1(age=23,gender="Female",name="Marry") #关键字参数可换顺序
user_info1("Jack",age=23,gender="Male") #位置参数和关键字参数，位置参数必须在关键字参数前
print()

# 缺省参数(默认值)
def user_info2(name,age,gender="Male"): # 设置默认值的参数需统一放在无默认值参数的后面
    print(f"Hello {name},you are {age},you are {gender}")
user_info2("Jack",age=23)
user_info2("Marry",gender="Female",age=23)
print()

# 不定长参数(可变参数)，不确定调用时传递几个参数(不传参也可以)
def test_args(*args): # 位置不定长，*表示标记形参为位置不定长，形参名args可以改成其他名字
    print(f"args类型为：{type(args)},内容是：{args}")
test_args("Jack","Marry","Jhon",11,22,33)
def test_kwargs(**kwargs): # 关键字不定长，**表示标记形参为关键字不定长，形参名kwargs可以改成其他名字
    print(f"kwargs类型为：{type(kwargs)},内容是：{kwargs}")
test_kwargs(name="Marry",age=23,gender="Female")
print()

# 函数作为参数传递
def add(a,b): # 数据传递
    return a+b # 调用时，a和b的值可变，但对a和b的操作(逻辑)不变(加法)
def multiply(a,b):
    return a*b
def input_function(function): # 逻辑传递
    result = function(2,3) # 调用时数据不变(2和3)，对数据的操作(逻辑)可变(取决于传入的函数)
    print(f"结果为：{result}")
input_function(add) # 传入add函数
input_function(multiply) # 传入multiply函数
print()

# lambda匿名函数，默认自带return，只能临时用一次。lambda 传入参数:函数体(只能有一行代码)
input_function(lambda x,y: x+y)
input_function(lambda x,y: x*y)