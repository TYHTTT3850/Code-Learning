"""
函数装饰器本质上是一个高阶函数，它接收一个函数作为参数，返回一个新的函数，该新函数通常在调用原函数前后增加一些逻辑。
"""
def my_wrapper(func):
    """
    装饰器就是相当于是说定义了一个特殊的函数，其参数为函数。
    这个函数里面又有一个函数，用于拓展传入的函数，返回的就是拓展后的函数。
    调用时，原本的函数会被替换成拓展后的函数
    """
    def wrapped(*args, **kwargs): # 函数里面的函数
        print("Before calling function.") #拓展原函数的功能
        result = func(*args, **kwargs)
        print("After calling function.")
        return result
    return wrapped # 返回拓展后的函数

@my_wrapper
def say_hello():
    print("Hello, World!") # 未拓展的函数的功能

say_hello() # 调用时被替换为拓展后的函数
print(say_hello.__name__) # 函数的名字被替换

# 使用装饰器时，原函数的名字、文档等信息可能会被包装器覆盖。为了避免这种情况，可以用functools.wraps
import functools

def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Before calling function.") #拓展原函数的功能
        result = func(*args, **kwargs)
        print("After calling function.")
        return result
    return wrapper

@my_decorator
def greet():
    """This is a greeting function"""
    print("Hi!")
greet()
print(greet.__name__)  # 输出 greet（而不是 wrapper）
print(greet.__doc__)   # 输出 This is a greeting function
