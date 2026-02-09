"""
装饰器场景：
1、无参无返回值的函数
2、有参无返回值的函数
3、无参有返回值的函数
4、有参有返回值的函数
5、不定长参数的函数
"""

# =========================
# 1. 无参、无返回值
# =========================
def decorator_1(func):
    def wrapper():
        print("[1] before")
        func()
        print("[1] after")
    return wrapper


@decorator_1
def f1():
    print("f1 running")


# =========================
# 2. 有参、无返回值
# =========================
def decorator_2(func):
    def wrapper(x, y):
        print("[2] before")
        func(x, y)
        print("[2] after")
    return wrapper


@decorator_2
def f2(x, y):
    print(x + y)


# =========================
# 3. 无参、有返回值
# =========================
def decorator_3(func):
    def wrapper():
        print("[3] before")
        result = func()
        print("[3] after")
        return result
    return wrapper


@decorator_3
def f3():
    return 100


# =========================
# 4. 有参、有返回值
# =========================
def decorator_4(func):
    def wrapper(x, y):
        print("[4] before")
        result = func(x, y)
        print("[4] after")
        return result
    return wrapper


@decorator_4
def f4(x, y):
    return x * y


# =========================
# 5. 不定长参数（通用）
# =========================
def decorator_5(func):
    def wrapper(*args, **kwargs):
        print("[5] before")
        # wrapper 可以接收任意数量的参数
        # 但不会检查参数是否合理
        # 所以多传入一个参数时，这里会报错
        result = func(*args, **kwargs)
        print("[5] after")
        return result
    return wrapper


@decorator_5
def f5(a, b, c=0):
    return a + b + c


# =========================
# 测试调用
# =========================
if __name__ == "__main__":
    f1()
    print("-" * 30)

    f2(3, 4)
    print("-" * 30)

    print(f3())
    print("-" * 30)

    print(f4(5, 6))
    print("-" * 30)

    print(f5(1, 2, c=3))
