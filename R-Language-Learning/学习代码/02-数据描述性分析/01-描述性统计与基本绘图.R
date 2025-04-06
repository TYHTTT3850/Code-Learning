# 描述性统计

data <- read.table("exec0301.data")
data <- c(data$V1,data$V2,data$V3,data$V4,data$V5,data$V6,data$V7,data$V8,data$V9,data$V10)
mean_data <- mean(data)
var_data <- var(data)
sd_data <- sd(data)
cv_data <- sd_data / mean_data
range_data <- max(data) - min(data)
se_data <- sd_data / sqrt(length(data))

library("e1071")
skewness_data <- skewness(data)
kurtosis_data <- kurtosis(data)

cat("平均值：",mean_data,"\n")
cat("方差：",var_data,"\n")
cat("标准差：",sd_data,"\n")
cat("变异系数：",cv_data,"\n")
cat("极差：",range_data,"\n")
cat("标准误：",se_data,"\n")
cat("偏度：",skewness_data,"\n")
cat("峰度：",kurtosis_data,"\n")

years <- c(rep(18,times = 110),rep(19,110),rep(20,100),rep(21,90),rep(22,90))
mean_years <- mean(years)

# 绘图

## 直方图
svg("直方图.svg", width = 8, height = 6)
hist(data,breaks = 10,
     probability = TRUE,
     col = "lightblue",
     border = "black",
     main = "Histogram of x",
     xlab = "value",
     ylab = "density")
lines(density(data), col = "blue", lwd = 2)
curve(dnorm(x, mean = mean(data), sd = sd(data)), col = "red",add = TRUE)
dev.off()

## 经验分布图
svg("经验分布图.svg",width = 8,height = 6)
plot(ecdf(data),col = "lightblue",verticals = TRUE,do.points = FALSE,main = "ecdf(x)")
curve(pnorm(x,mean = mean(data),sd = sd(data)),col = "red",add = TRUE)
dev.off()

## QQ图
svg("QQ图.svg",width = 8,height = 6)
qqnorm(data,
       main = "Q-Q plot",
       col = "blue")
dev.off()

## 茎叶图
stem(data)

## 箱线图
svg("箱线图.svg",width = 8,height = 6)
boxplot(data)
dev.off()