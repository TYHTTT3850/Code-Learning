"""
模块导入常用方法:
模块导入一般写在代码开头
    import 模块名
    from 模块名 import 类、变量、方法等
    from 模块名 import *
    import 模块名 as 别名
    from 模块名 import 功能名 as 别名
"""
# import time # 导入整个time模块，可使用time模块的全部功能
# print("hello world")
# time.sleep(5) # 使用模块内的功能，模块名.功能名
# print("Hello World")

# from time import sleep # 只导入time模块的sleep功能
# print("hello world")
# sleep(5) # 调用时直接使用相应的功能，而不是模块名.功能名
# print("Hello World")

# from time import * #导入time模块的全部功能
# print("hello world")
# sleep(5) # 调用时直接使用相应的功能，而不是模块名.功能名
# print("Hello World")

# import time as t # 以别名t导入整个time模块
# print("hello world")
# t.sleep(5) # 使用模块内的功能，别名.功能名
# print("Hello World")

# from time import sleep as sl # 以别名sl导入time模块的sleep功能
# print("hello world")
# sl(5) # 调用时以别名的形式直接使用相应的功能
# print("Hello World")
