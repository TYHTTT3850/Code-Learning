# 1、矩估计
binom_data <- scan("binom.data")
# 样本均值和样本方差
A1 <- mean(binom_data)
n <- length(binom_data)
M2 <- (n-1)/n*var(binom_data)
# 矩估计法计算 n 和 p
n_hat <- A1^2 / (A1 - M2)
p_hat <- (A1-M2) / A1


# 2、概率密度(a+1)x^a 0<x<1
x <- c(0.1, 0.2, 0.9, 0.8, 0.7, 0.7, 0.6, 0.5)
n <- length(x)
# 样本均值
x_bar <- mean(x)
# 矩估计
alpha_mom <- (2 * x_bar - 1) / (1 - x_bar)
# 极大似然估计
alpha_mle <- -n / sum(log(x)) - 1

# 3、指数分布极大似然估计
# 组中值和频数
x <- c(5, 15, 25, 35, 45, 55, 65)
f <- c(365, 245, 150, 100, 70, 45, 25)
# 总样本数
n <- sum(f)
# 加权样本均值
x_bar <- sum(f * x) / n
# 极大似然估计
lambda_hat <- 1 / x_bar

# 4、泊松分布极大似然估计
x <- 0:6
f <- c(17, 20, 10, 2, 1, 0, 0)
n <- sum(f)
lambda_hat <- sum(x * f) / n