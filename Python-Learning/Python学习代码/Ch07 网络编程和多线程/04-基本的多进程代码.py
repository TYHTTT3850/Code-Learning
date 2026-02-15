import multiprocessing
import time
# 1、定义函数
def coding():
    for i in range(1,11):
        time.sleep(0.1) # 模拟耗时操作
        print(f"正在第{i}遍编程", end="\n")

def listening():
    for i in range(1,11):
        time.sleep(0.1) # 模拟耗时操作
        print(f"正在第{i}遍听歌", end="\n")

# 2、创建进程对象
if __name__ == "__main__": # 必须这么写，不然报错
    p1 = multiprocessing.Process(target=coding)
    p2 = multiprocessing.Process(target=listening)

    # 3、启动进程
    p1.start()
    p2.start()