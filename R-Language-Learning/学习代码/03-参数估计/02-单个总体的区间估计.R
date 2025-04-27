# 1、均值估计：t统计量
# 样本数据
ages <- c(32, 50, 40, 24, 33, 44, 45, 48, 44, 47, 31, 36,
          39, 46, 45, 39, 38, 45, 27, 43, 54, 36, 34, 48,
          23, 36, 42, 34, 39, 34, 35, 42, 53, 28, 49, 39)
# 样本均值
mean_age <- mean(ages)
# 样本标准差
sd_age <- sd(ages)
# 样本大小
n <- length(ages)
# t临界值（双侧，置信水平90%，自由度n-1）
alpha <- 0.10
t_critical <- qt(1 - alpha/2, df = n - 1)
# 置信区间半宽
margin_error <- t_critical * sd_age / sqrt(n)
# 置信区间
lower_bound <- mean_age - margin_error
upper_bound <- mean_age + margin_error
# 输出结果
cat("90%置信区间为：(", lower_bound, ",", upper_bound, ")\n")

# 2、均值估计：z统计量
# 数据
times <- c(0, 1, 2, 3, 4, 5, 6)
freqs <- c(7, 10, 12, 8, 3, 2, 0)
# 总样本量
n <- sum(freqs)
# 样本均值（λ的估计）
lambda_hat <- sum(times * freqs) / n
# 标准误差
se <- sqrt(lambda_hat / n)
# z临界值（95%置信水平）
z_critical <- qnorm(1 - 0.05/2)
# 置信区间
lower_bound <- lambda_hat - z_critical * se
upper_bound <- lambda_hat + z_critical * se
# 输出
cat("λ的点估计为：", lambda_hat, "\n")
cat("95%置信区间为：(", lower_bound, ",", upper_bound, ")\n")

# 3、单侧区间估计
# 样本数据
pulse <- c(54, 67, 68, 78, 70, 66, 67, 70, 65, 69)
# 样本均值和标准差
mean_pulse <- mean(pulse)
sd_pulse <- sd(pulse)
n <- length(pulse)
# 标准误差
SE <- sd_pulse / sqrt(n)
# t临界值（双侧95%）与单侧95%
alpha <- 0.05
t_critical_two_sided <- qt(1 - alpha/2, df = n-1)  # 双侧95%
t_critical_one_sided <- qt(1 - alpha, df = n-1)  # 单侧95%
# 95%双侧置信区间
ci_lower <- mean_pulse - t_critical_two_sided * SE
ci_upper <- mean_pulse + t_critical_two_sided * SE
# 95%单侧置信区间（上限）
single_side_upper <- mean_pulse + t_critical_one_sided * SE
# 单侧t统计量
t_statistic <- (mean_pulse - 72) / SE
# 输出结果
cat("点估计（均值） =", mean_pulse, "\n")
cat("95%双侧置信区间 = (", ci_lower, ",", ci_upper, ")\n")
cat("95%单侧置信区间：(-∞,", single_side_upper, ")\n")
cat("单侧t统计量 =", t_statistic, "\n")
cat("单侧临界值 =", t_critical_one_sided, "\n")
# 判断单侧检验
if (t_statistic < t_critical_one_sided) {
  cat("结论：有显著证据表明患者脉搏次数低于正常水平。\n")
} else {
  cat("结论：没有足够证据表明患者脉搏次数低于正常水平。\n")
}

# 4、估计单侧置信下限
# 样本数据
lifespan <- c(1067, 919, 1196, 785, 1126, 936, 918, 1156, 920, 948)
# 样本均值和标准差
mean_lifespan <- mean(lifespan)
sd_lifespan <- sd(lifespan)
n <- length(lifespan)
# 标准误差
SE <- sd_lifespan / sqrt(n)
# t临界值（单侧95%）
alpha <- 0.05
t_critical_one_sided <- qt(1 - alpha, df = n-1)  # 单侧95%
# 单侧置信下限
ci_lower <- mean_lifespan - t_critical_one_sided * SE
# 输出
cat("样本均值 =", mean_lifespan, "\n")
cat("样本标准差 =", sd_lifespan, "\n")
cat("标准误差 =", SE, "\n")
cat("单侧95%置信下限 =", ci_lower, "\n")
