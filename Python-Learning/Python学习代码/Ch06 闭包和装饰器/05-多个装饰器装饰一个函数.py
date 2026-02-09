import functools

def make_bold(func):
    """装饰器1：将结果加粗"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("--- 正在执行 make_bold ---")
        return "<b>" + func(*args, **kwargs) + "</b>"
    return wrapper

def make_italic(func):
    """装饰器2：将结果变成斜体"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("--- 正在执行 make_italic ---")
        return "<i>" + func(*args, **kwargs) + "</i>"
    return wrapper

"""
当使用多个装饰器时，它们的执行顺序是：
从下往上装饰，从上往下执行
即最靠近函数的装饰器最先被应用，但最外层的装饰器代码最先开始运行。
"""
@make_bold
@make_italic
def hello(name):
    print("--- 正在执行 hello 函数 ---")
    return f"Hello, {name}"

if __name__ == "__main__":
    result = hello("GitHub Copilot")
    print("\n最终结果:", result)

"""
    hello 函数被两个装饰器包裹：
    1. 装饰器的应用顺序（定义时）是从下往上：
       - 先应用 make_italic → 生成新函数
       - 再应用 make_bold → 生成最终函数
    2. 实际函数调用顺序（运行时）是从外向内：
       - 调用最外层 wrapper (也就是 make_bold)
       - wrapper 内部调用 func(*args, **kwargs) → 进入 make_italic 所装饰后的函数
       - make_italic 再调用原 hello 函数
    3. 返回值顺序：
       - hello 返回内容 → 被 make_italic 包裹 → 再被 make_bold 包裹
       - 所以最终 HTML 结果是 <b><i>内容</i></b>
    """