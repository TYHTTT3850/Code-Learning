try: # 可能会报错的代码
    file1 = open("异常捕获与处理.txt","r",encoding="utf-8") #会出现报错FileNotFoundError
except Exception as e: # 出现报错后的操作
    print("出现异常")
    print(e)
    file1 = open("异常捕获与处理.txt", "w", encoding="utf-8")
else: # 没有报错的操作
    print("没有异常")
finally: # 无论是否报错，都执行的操作
    file1.close()
# except可以指定一种或多种错误类型。except(错误类型1,错误类型2,...,错误类型N):
# 没有指定错误类型或指定错误类型为Exception时捕获所有错误
# except(错误类型1,错误类型2,...,错误类型N) as 错误对象名:
# 错误对象即为具体错误信息