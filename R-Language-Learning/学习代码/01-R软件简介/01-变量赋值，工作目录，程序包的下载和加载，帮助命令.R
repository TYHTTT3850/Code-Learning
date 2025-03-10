# 变量赋值
x <- c(-1,0,2);
y <- c(3,8,2);
z <- 2*x+y+1;

# 获取工作目录
getwd();
# 使用setwd("工作目录")切换工作目录，注意使用/来表示层级关系

# 加载程序包

library("foreign");

read.spss("educ_scores.sav");

# 设定CRAN镜像，安装程序包

options(repos=structure(c(CRAN="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")))

install.packages("e1071")

library("e1071")

# 使用帮助命令查看函数的使用方法

help("t.test") #只需要函数名，不需要括号
