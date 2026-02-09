import functools

def repeat(n=1):
    """
    因为装饰器最多只能有一个参数(被装饰的函数)，所以我们需要在外层函数中接收装饰器参数，并在内层函数中使用这些参数来控制装饰器的行为。所以使用带有参数的装饰器，其实就是在装饰器外面再套一层函数，使用该函数接收参数，返回装饰器。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = None
            print(f"--- 开始执行 {func.__name__}，共重复 {n} 次 ---")
            for i in range(n):
                print(f"第 {i+1} 次调用:")
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

# 使用带参数的装饰器
@repeat(n=3)
def greet(name):
    print(f"Hello, {name}!")

# 这个写法就相当于是 greet = repeat(n=3)(greet)

if __name__ == "__main__":
    greet("World")

"""
带参数的装饰器的执行过程：

1. 执行 repeat(n=3)
   → 返回 decorator 函数(闭包中保存 n=3)

2. 创建原始 greet 函数对象

3. 执行 decorator(greet)
   → 返回 wrapper 函数(闭包中保存 func=greet, n=3)

4. 用 wrapper 替换函数名
   → greet = wrapper

5. 调用阶段：
   greet("World") 实际执行的是 wrapper("World")
   wrapper 内部调用原始 greet 共 3 次
"""

