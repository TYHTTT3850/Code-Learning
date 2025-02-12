from sympy import *

"""--------------------创建符号变量--------------------"""
x = symbols('x') #x表示符号变量名，'x'是符号变量值。
y,z = symbols('y z') #一次创建多个符号变量
m0,m1,m2,m3 = symbols('m0:4')
var_v = symbols('v') #符号变量名与符号变量值不一定要相等。
x = sin(1)

"""--------------------获取浮点近似值--------------------"""
print("--------------------获取浮点近似值--------------------")
#.evalf(),.n()。符号变量类方法，默认精度是小数点后15位，可以通过调整参数获得任何想要的精度
print('x=',x)
print("x=",x.evalf())
print("x=",x.n(16))
print(f"Pi的两种显示格式：{pi},{pi.evalf(3)}") #这里不能使用.n()方法
print()

"""--------------------创建符号表达式--------------------"""
print("--------------------创建符号表达式--------------------")
expr1 = y*sin(y**2)
expr2 = y**2+sin(y)*cos(y)+sin(z)
print("expr1 =",expr1)
print("expr2 =",expr2)

#.subs()方法。带入一个符号变量的值
print("y=5时，expr1=",expr1.subs(y,5))
print("y=2，z=3时，expr2=",expr2.subs({y:2,z:3}))
print("y=2，z=3时，expr2=",expr2.subs({y:2,z:3}).n())
print()

"""--------------------有理表达式的合并与拆分--------------------"""
print("--------------------有理表达式的合并与拆分--------------------")
expr3 = 1/(m0+1) + 1/(m1+2)
expr4 = (2*m0**2+3*m0+4)/(m0+1)
print("expr3表达式合并：",together(expr3))
print("expr4表达式拆分：",apart(expr4))
print(simplify(expr4)) #.simplify()方法没有效果