# 秦九韶算法：一种简化多项式运算的算法。
def qin_jiu_shao(list,x):
    result=list[0]
    for a in list[1:]:
        result=result*x+a
    return result
x = int(input())
list=[4,2,3.5,-2.6,1.7,-0.8] #可用循环语句依次输入多项式系数 
print(qin_jiu_shao(list,x))
