data(pressure)

# plot绘图样式
svg("pressuer数据集绘图.svg", width = 15, height = 15,family = "SimSun")
par(mfrow = c(3, 3)) #设置子图布局3*3
plot(pressure$temperature,pressure$pressure,xlab = "温度",ylab = "压力",main = "(1)绘出散点图")
plot(pressure$temperature, pressure$pressure, type = "l", xlab = "温度", ylab = "压力", main = "(2)只画线不画点")
plot(pressure$temperature, pressure$pressure, type = "b", xlab = "温度", ylab = "压力", main = "(3)同时绘点和线")
plot(pressure$temperature, pressure$pressure, type = "c", xlab = "温度", ylab = "压力", main = "(4)只画(3)中的线")
plot(pressure$temperature, pressure$pressure, type = "o", xlab = "温度", ylab = "压力", main = "(5)同时绘点和画线且线穿过点")
plot(pressure$temperature, pressure$pressure, type = "h", xlab = "温度", ylab = "压力", main = "(6)绘出点到横轴的坚线")
plot(pressure$temperature, pressure$pressure, type = "s", xlab = "温度", ylab = "压力", main = "(7)绘出阶梯图(先横再纵)")
plot(pressure$temperature, pressure$pressure, type = "S", xlab = "温度", ylab = "压力", main = "(8)绘出阶梯图(先纵再横)")
plot(pressure$temperature, pressure$pressure, type = "n", xlab = "温度", ylab = "压力", main = "(9)作一幅空图")
dev.off()

# boxplot语句多组数据绘制箱线图
# 每组数据
type1 <- c(2, 4, 3, 2, 4, 7, 7, 2, 2, 5, 4)             # 11 个
type2 <- c(5, 6, 8, 5, 10, 7, 12, 12, 6, 6)             # 10 个
type3 <- c(7, 11, 6, 6, 7, 9, 5, 5, 10, 6, 3, 10)       # 12 个

df1 <- data.frame("type"=c(rep("菌型1",times=length(type1)),rep("菌型2",times=length(type2)),rep("菌型3",times=length(type3))),"days"=c(type1,type2,type3))

svg("小鼠存活.svg",width=8,height=6,family = "SimSun")
boxplot(days ~ type,data = df1, main = "小白鼠接种不同菌型后的存活日数", xlab = "菌型", ylab = "存活日数", col = c("lightblue", "lightgreen", "lightpink"))
dev.off()

# plot语句绘制多组箱线图
# 导出为SVG
svg("plot语句_小鼠存活.svg", width = 8, height = 6, family = "SimSun")
box_list <- split(df1$days, df1$type)

plot(1:3,
     type = "n",
     xaxt = "n",
     ylim = c(0, max(df1$days) + 2),
     xlab = "菌型",
     ylab = "存活天数",
     main = "小白鼠存活天数箱线图（plot语句）")

axis(1, at = 1:3, labels = names(box_list))

for (i in 1:3) {
  boxplot(box_list[[i]],
          at = i,
          add = TRUE,
          col = rainbow(3)[i],
          boxwex = 0.5)
}
dev.off()

# 阵列式散点图
# 加载iris数据集
data(iris)

# 直方图函数（用于对角线）
panel.hist <- function(x, ...) {
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5))
  hist(x, col = "lightblue", border = "white", probability = TRUE, add = TRUE)
}

# 相关系数函数（用于左下角）
panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...) {
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if (missing(cex.cor)) cex.cor <- 0.8 / strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor)
}

# 平滑函数（用于右上角）
panel.smooth <- function(x, y, col = par("col"), bg = NA, pch = par("pch"),
                         cex = 1, col.smooth = "red", span = 2/3, ...) {
  points(x, y, pch = pch, col = col, bg = bg, cex = cex)
  ok <- is.finite(x) & is.finite(y)
  if (any(ok))
    lines(stats::lowess(x[ok], y[ok], f = span), col = col.smooth, ...)
}


svg("鸢尾花阵列式散点图.svg",width=8,height=6,family = "SimSun")
# 创建阵列式散点图
pairs(iris[1:4],
      upper.panel = panel.smooth,   # 右上角：散点 + 平滑线
      lower.panel = panel.cor,      # 左下角：相关系数
      diag.panel  = panel.hist,     # 对角线：直方图
      main = "鸢尾花特征的阵列式散点图")
dev.off()

# 绘制协同图
# 加载trees数据集
data(trees)
panel.Lm.R <- function(x, y, ...) {
  points(x, y, pch = 1, col = "black")      # 画散点
  abline(lm(y ~ x), col = "black", lwd = 1.5)  # 加回归线
}

svg("协同图.svg",width=8,height=6,family = "SimSun")
coplot(Volume ~ Height | Girth, data = trees,
       panel = panel.Lm.R,
       number = 6,
       overlap = 0.5,
       show.given = TRUE,
       xlab = "Height", ylab = "Volume")
dev.off()
