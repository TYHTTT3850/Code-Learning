###########################################################
# 输入数据
n <- 400        # 样本容量
x <- 57         # 样本中老年人数
# 样本比例
p_hat <- x / n
# 置信水平 95%，查标准正态分布临界值 Z
z <- 1.96
# 标准误差
se <- sqrt(p_hat * (1 - p_hat) / n)
# 置信区间
lower <- p_hat - z * se
upper <- p_hat + z * se
# 输出结果
cat("老年人比例的95%置信区间为：(", round(lower, 4), ",", round(upper, 4), ")\n")

###########################################################
jia <- c(97,90,94,79,78,87,83,89,76,84,
         83,84,76,82,85,85,91,72,86,70,
         91,87,73,92,64,74,88,88,74,73)
yi <- c(64,85,72,64,74,93,70,79,79,75,
        66,83,74,70,82,82,75,78,99,57,
        91,78,87,93,89,79,84,65,78,66,
        84,85,85,84,59,62,91,83,80,76)
# 计算均值和标准差
mean_jia <- mean(jia)
mean_yi <- mean(yi)
sd_jia <- sd(jia)
sd_yi <- sd(yi)
n_jia <- length(jia)
n_yi <- length(yi)
# 差的置信区间公式（假设方差不等）
t_result <- t.test(jia, yi, var.equal = FALSE, conf.level = 0.95)
# 输出结果
cat("甲平均分：", round(mean_jia, 2), "\n")
cat("乙平均分：", round(mean_yi, 2), "\n")
cat("平均分差的95%置信区间：", round(t_result$conf.int[1], 2), "到", round(t_result$conf.int[2], 2), "\n")


###########################################################
# 输入数据
jia <- c(140, 137, 136, 140, 145, 148, 140, 135, 144, 141)
yi  <- c(135, 118, 115, 140, 128, 131, 130, 115, 131, 125)
# 差值
diff <- jia - yi
# 计算均值和标准差
mean_diff <- mean(diff)
sd_diff <- sd(diff)
n <- length(diff)
# 临界 t 值 (自由度 = n - 1)
alpha <- 0.05
t_crit <- qt(1 - alpha/2, df = n - 1)
# 置信区间计算
se <- sd_diff / sqrt(n)
lower <- mean_diff - t_crit * se
upper <- mean_diff + t_crit * se
# 使用 t.test 来直接计算
t_result <- t.test(jia, yi, paired = TRUE, conf.level = 0.95)
# 输出结果
cat("差值均值：", round(mean_diff, 2), "\n")
cat("差值标准差：", round(sd_diff, 2), "\n")
cat("95%置信区间：(", round(lower, 2), ",", round(upper, 2), ")\n")
# 或输出t检验内置置信区间
cat("t.test()计算置信区间：", round(t_result$conf.int[1], 2), "到", round(t_result$conf.int[2], 2), "\n")

###########################################################
# 数据输入
jia <- c(0.143, 0.142, 0.143, 0.137)
yi  <- c(0.140, 0.142, 0.136, 0.138, 0.140)
# 计算均值和标准差
mean_jia <- mean(jia)
mean_yi  <- mean(yi)
sd_jia <- sd(jia)
sd_yi  <- sd(yi)
n_jia <- length(jia)
n_yi  <- length(yi)
# 输出均值和标准差
cat("甲组均值：", round(mean_jia, 5), "，标准差：", round(sd_jia, 5), "\n")
cat("乙组均值：", round(mean_yi, 5), "，标准差：", round(sd_yi, 5), "\n")
# t检验（方差相等）
t_result <- t.test(jia, yi, var.equal = TRUE, conf.level = 0.95)
# 输出置信区间结果
cat("μ1 - μ2 的 95% 置信区间：", round(t_result$conf.int[1], 5), "到", round(t_result$conf.int[2], 5), "\n")

###########################################################
# 数据输入
jia <- c(140, 137, 136, 140, 145, 148, 140, 135, 144, 141)
yi  <- c(135, 118, 115, 140, 128, 131, 130, 115, 131, 125)

# 样本信息
n1 <- length(jia)
n2 <- length(yi)
var_jia <- var(jia)
var_yi  <- var(yi)
F0 <- var_jia / var_yi

# 置信水平
alpha <- 0.05
F_lower <- qf(alpha / 2, df1 = n1 - 1, df2 = n2 - 1)
F_upper <- qf(1 - alpha / 2, df1 = n1 - 1, df2 = n2 - 1)
CI_lower <- F0 / F_upper
CI_upper <- F0 / F_lower

cat("【方差比检验】\n")
cat("甲方差：", round(var_jia, 4), " 乙方差：", round(var_yi, 4), "\n")
cat("方差比（甲/乙）：", round(F0, 4), "\n")
cat("方差比的95%置信区间：(", round(CI_lower, 4), ",", round(CI_upper, 4), ")\n")

# 判断是否采用等方差检验
equal_var <- ifelse(CI_lower <= 1 && 1 <= CI_upper, TRUE, FALSE)
cat("是否可认为等方差：", equal_var, "\n\n")

cat("【均值差的置信区间估计】\n")
if (equal_var) {
  t_result <- t.test(jia, yi, var.equal = TRUE, conf.level = 0.95)
} else {
  t_result <- t.test(jia, yi, var.equal = FALSE, conf.level = 0.95)
}
cat("甲平均产量：", mean(jia), " 乙平均产量：", mean(yi), "\n")
cat("μ1 - μ2 的 95% 置信区间：", round(t_result$conf.int[1], 2), "到", round(t_result$conf.int[2], 2), "\n")
