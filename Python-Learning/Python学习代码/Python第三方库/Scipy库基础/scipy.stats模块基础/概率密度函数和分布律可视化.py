from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rc('text', usetex=True)

# 一个界面中4个不同的Γ分布密度函数
fig1 = plt.figure()
axe1_1 = fig1.add_subplot(111)
x = np.linspace(0, 15, 101)
axe1_1.plot(x, stats.gamma.pdf(x,a=4,loc=0,scale=2),'r*-',label="$\\alpha=4,\\beta=2$")
axe1_1.plot(x, stats.gamma.pdf(x,a=4,loc=0,scale=1),'bp-',label="$\\alpha=4,\\beta=1$")
axe1_1.plot(x, stats.gamma.pdf(x,a=4,loc=0,scale=0.5),'.k-',label="$\\alpha=4,\\beta=0.5$")
axe1_1.plot(x, stats.gamma.pdf(x,a=2,loc=0,scale=0.5),'>g-',label="$\\alpha=2,\\beta=2$")
axe1_1.legend(loc='upper right')
axe1_1.set_xlabel('$x$')
axe1_1.set_ylabel('$f(x)$',rotation=0)
fig1.tight_layout()

# 四个子界面中绘制四个不同的正态分布密度函数
fig2 = plt.figure()
x = np.linspace(-7, 7, 141)
gs = gridspec.GridSpec(2, 2) #创建一个GridSpec对象
axe2_1 = fig2.add_subplot(gs[0, 0])
axe2_2 = fig2.add_subplot(gs[0, 1])
axe2_3 = fig2.add_subplot(gs[1, 0])
axe2_4 = fig2.add_subplot(gs[1, 1])
axe2_1.plot(x,stats.norm.pdf(x,loc=-1,scale=0.5),label='$a=-1$\n$\\sigma=0.5$')
axe2_2.plot(x,stats.norm.pdf(x,loc=-1,scale=1),label='$a=-1$\n$\\sigma=1$')
axe2_3.plot(x,stats.norm.pdf(x,loc=0,scale=0.5),label='$a=0$\n$\\sigma=0.5$')
axe2_4.plot(x,stats.norm.pdf(x,loc=0,scale=1),label='$a=0$\n$\\sigma=1$')
axe2_1.legend(loc='upper right')
axe2_2.legend(loc='upper right')
axe2_3.legend(loc='upper right')
axe2_4.legend(loc='upper right')
fig2.tight_layout()

# 两种方法绘制二项分布X~b(n,p)的分布律的"火柴杆"图。n=5,p=0.4。
n,p = 5,0.4
fig3 = plt.figure()
x = np.arange(6)
axe3_1 = fig3.add_subplot(121)
axe3_2 = fig3.add_subplot(122)
y = stats.binom.pmf(x,p=p,n=n)
# 第一种方式。点和垂直线组合
axe3_1.plot(x,y,'ko')
axe3_1.vlines(x,0,y,'k',
              label=f'$n={n}$\n$p={p}$') #.vlines(x,ymin,ymax)
axe3_1.legend(loc='upper right')

# 第二种方式
axe3_2.stem(x,y)
"""use_line_collection=True 的作用是优化绘图性能，特别是在绘制大量线条时。"""

plt.show()