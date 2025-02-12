import random
import math
for i in range(150):
    a = 1000*random.random()
    b = 1000*random.random()
    c = random.randint(-999,1000)
    d = random.randint(-999,1000)
    e = random.randint(1,101)
    f = random.randint(1,101)
    g = random.randint(-10,11)
    if (e-f) != 0 and g != 0:
        tmp = math.gcd(g,e-f)
        print("-%.2f + %.2f + [(%d) × (%d)] + [%d ÷ (%d-%d)] =" %(a,b,c,d,g,e,f))
        print("答案：%.2f + (%d/%d)\n"%((-a+b+c*d),(g/tmp),((e-f)/tmp)))
input("按Enter键退出")