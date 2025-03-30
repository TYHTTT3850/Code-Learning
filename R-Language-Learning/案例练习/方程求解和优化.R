# 求解极小值点和极大值点
# 定义函数
f <- function(x) {
  return(exp(-x^2) * (x + sin(x)))
}

# 计算导数
library(Deriv)
f_prime <- Deriv(f, x = "x")   # 一阶导数
f_double_prime <- Deriv(f_prime, x = "x")  # 二阶导数

# 求解 f'(x) = 0
library(rootSolve)
root_result <- uniroot.all(f_prime, c(-2, 2))

# 计算二阶导数值，判断极值类型
for (x in root_result) {
  second_derivative_value <- f_double_prime(x)
  if (second_derivative_value > 0) {
    cat("极小值点：x =", x, ", f(x) =", f(x), "\n")
  } else if (second_derivative_value < 0) {
    cat("极大值点：x =", x, ", f(x) =", f(x), "\n")
  }
}

# 直接求解极大值点和极小值点
extreames <- c(optimise(f,c(-2,2)),optimise(f,c(-2,2),maximum = TRUE))

# 求解非线性方程组
funs <- function(x){
  return(c(x[1]+0.7*sin(x[1])-0.2*cos(x[2]),x[2]-0.7*cos(x[1])+0.2*sin(x[2])))
}
library("nleqslv")
sol <- nleqslv(x=c(0.5,-2),fn=funs)

# 求解多元函数的极小值点
f1 <- function (x){
  return( (-13+x[1]+((5-x[2])*x[2]-2)*x[2])^2+(-29+x[1]+((x[2]+1)*x[2]-14)*x[2]) )
}
min <- nlm(f1,c(-0.5,2))

program <- function(n){
  count <- 0
  while (n != 1){
    if (n <= 0){
      cat("计算终止")
      break
    }
    else if(n %% 2 == 0){
      n <- n/2
      count <- count + 1
    }
    else{
      n <- 3*n+1
      count <- count + 1
    }
  }
  return(count)
}

count <- program(10)
cat(count)