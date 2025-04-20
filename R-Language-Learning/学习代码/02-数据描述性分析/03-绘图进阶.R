# 1、绘制协同图
data(rock)
svg("第一题.svg",height = 8,width = 6)
coplot(perm ~ shape | area * peri, data = rock,
       xlab = "Shape", ylab = "Perm",
       show.given = c(TRUE, TRUE))  # 显示 area 和 peri 的分组条件
dev.off()

# 多元函数绘图
# 定义网格
x <- seq(-2, 3, by = 0.05)
y <- seq(-1, 7, by = 0.05)
grid <- expand.grid(x = x, y = y)

# 计算 z 值
z <- with(grid, x^4 - 2*x^2*y + x^2 - 2*x*y + 2*y^2 + (9/2)*x - 4*y + 4)

# 将 z 变为矩阵，方便绘图
z_matrix <- matrix(z, nrow = length(x), ncol = length(y))

# 2.1、三维网格曲面图
svg("第二题三维图.svg",height = 6,width = 8)
persp(x, y, z_matrix,
      theta = 45, phi = 30,      # 设置观察角度
      expand = 0.6,              # 缩放比例
      col = "lightblue",         # 表面颜色
      xlab = "x", ylab = "y", zlab = "z",
      ticktype = "detailed",     # 精细坐标轴
      main = "3D Surface of z = f(x, y)")
dev.off()

# 2.2、二维等值线图
svg("第二题等值线.svg",width = 8,height = 6)
contour(x, y, z_matrix,
        levels = c(0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100),
        xlab = "x", ylab = "y",
        main = "Contour Plot of z = f(x, y)",
        col = "blue", lwd = 1.5)
dev.off()

# 3、绘制星图
applicant <- read.table("applicant.data")
svg("第三题(1).svg",height = 6,width = 8)
stars(applicant[, -1])
dev.off()

# 4、自定义星图的轴
applicant2 <- data.frame(G1 = (applicant$SC + applicant$LC + applicant$SMS + applicant$DRV + applicant$AMB + applicant$GSP + applicant$POT)/7, G2 = (applicant$FL + applicant$EXP + applicant$SUIT)/3, G3 = (applicant$LA + applicant$HON + applicant$KJ)/3, G4 = applicant$AA, G5 = applicant$APP)

svg("第三题(2).svg",height = 6,width = 8)
stars(applicant2)
dev.off()

# 绘制调和曲线
unison <- function(x) {
    if (is.data.frame(x) == TRUE)
        x <- as.matrix(x)
    t <- seq(-pi, pi, pi/30)
    m <- nrow(x)
    n <- ncol(x)
    f <- array(0, c(m, length(t)))
    for (i in 1:m) {
        f[i, ] <- x[i, 1]/sqrt(2)
        for (j in 2:n) {
            if (j%%2 == 0)
                f[i, ] <- f[i, ] + x[i, j] * sin(j/2 * t) else f[i, ] <- f[i, ] + x[i, j] * cos(j%/%2 * t)
        }
    }
    plot(c(-pi, pi), c(min(f), max(f)), type = "n", main = "The Unison graph of Data",
        xlab = "t", ylab = "f(t)")
    for (i in 1:m) lines(t, f[i, ], col = i)
}
svg("第四题.svg",width = 8,height = 6)
unison(applicant2)
dev.off()
