### 向量：R语言的基本数据结构，由相同类型的元素组成。

v <- c(10, 20, 30, 40, 50)  #创建数值型向量
v2 <- c("A", "B", "C") #字符型向量

# seq()函数
v3 <- seq(0,10,by=3) #by表示按照指定间隔生成元素。
v4 <- seq(0,10,length.out = 4) #length.out表示等间隔地生成指定数量的元素。

# rep()函数
x <- rep(1,times=5) #指定值重复指定次

length(v)      #获取向量长度
sum(v)         #计算总和
mean(v)        #计算均值
sort(v)        #排序
unique(v)      #去重
names(v) <- c("A", "B", "C", "D", "E")  #给向量命名
v[2]       #按索引访问，返回20
v["C"]     #按名称访问，返回30
v[c(2, 4)] #多索引访问，返回 20 40
v[-3]      #负索引访问，去掉第 3 个元素，返回 10 20 40 50
v[v > 25]  #逻辑条件访问，返回 30 40 50

# 向量运算
v1 <- c(1, 2, 3)
v2 <- c(4, 5, 6)
# 算术运算
v1 + v2  # 1+4, 2+5, 3+6
v1 - v2  # 1-4, 2-5, 3-6
v1 * v2  # 1*4, 2*5, 3*6
v1 / v2  # 1/4, 2/5, 3/6
v1 ^ v2  # 1^4, 2^5, 3^6
# 逻辑运算
v1 > v2   # 逐个比较
v1 == v2  # 判断是否相等
v1 != v2  # 判断是否不等
# 和标量运算
v1 + 2   # 每个元素 +2
v1 * 3   # 每个元素 *3
# 内积和外积
crossprod(v1, v2) #点乘，内积
outer(v1,v2) #叉乘，外积


### 因子：用于存储分类数据，存储的值是类别(levels)。

f <- factor(c("Male", "Female", "Male", "Female", "Male"))
levels(f)   #查看因子的类别
table(f)    #统计每个类别的频数


### 矩阵：二维数组，所有元素类型相同。

m <- matrix(1:9, nrow = 3, ncol = 3) #默认按列填充
m2 <- matrix(1:9, nrow = 3, ncol = 3, byrow = TRUE)
rownames(m) <- c("Row1", "Row2", "Row3")  #设置行名
colnames(m) <- c("Col1", "Col2", "Col3")  #设置列名
m3 <- matrix(1:9, nrow = 3, ncol = 3,
             dimnames = list(c("A", "B", "C"), c("X", "Y", "Z"))) #创建矩阵时直接指定行名和列名

# 利用其他矩阵创建矩阵
A <- matrix(1:20,nrow=4,ncol=5)
B <- matrix(1:20,nrow=4,ncol=5,byrow=TRUE)
C <- matrix(A[1:3,1:3],3,3)
D <- matrix(B[,-3],4,4)

# 利用向量创建矩阵
x <- c(1,3,5,7,9)
X <- matrix(c(rep(1,5),x,x^2),nrow=5,ncol=3) #第一列全为1，第二列为向量x，第三列为x^2

dim(m)       #查看维度
t(m)         #转置
rowSums(m)   #计算每行的总和
colMeans(m)  #计算每列的均值
rownames(m)  #获取行名
colnames(m)  #获取列名
rownames(m) <- NULL  # 删除行名
colnames(m) <- NULL  # 删除列名

# 矩阵运算
A <- matrix(1:9, nrow = 3, byrow = TRUE)
B <- matrix(9:1, nrow = 3, byrow = TRUE)
# 逐元素运算
A + B  # 逐元素加
A - B  # 逐元素减
A * B  # 逐元素乘
A / B  # 逐元素除
A ^ 2  # 每个元素平方
# 矩阵乘法
A %*% B  # 矩阵乘法（行×列）
# 矩阵转置
t(A)  # 返回转置矩阵
# 矩阵求逆
M <- matrix(c(2, -1, 0, 1), nrow = 2)
solve(M)  # 返回逆矩阵
# 行/列求和 & 均值
rowSums(A)  # 计算每行的和
colSums(A)  # 计算每列的和
rowMeans(A) # 计算每行的均值
colMeans(A) # 计算每列的均值
# 行列式
det(A)  # 计算行列式


### 列表：可以存储不同类型的数据对象。键值对的的形式存储，键=值。

lst <- list(name = "Alice", age = 25, scores = c(90, 85, 88))
lst$name       #访问元素，返回 "Alice"
lst[["name"]]   #返回 "Alice"
lst["name"]    #返回一个子列表，而不是字符串 "Alice"
length(lst)    #列表长度


### 数据框：类似于表格，每列可以是不同的数据类型。每列都是向量，本质就是每列都是向量的列表

df <- data.frame(Name = c("A", "B", "C"), Age = c(25, 30, 35), Score = c(80, 90, 85))
dim(df)       #获取行列数
head(df)      #查看前6行
str(df)       #查看数据结构
summary(df)   #统计信息
df$Age        #访问 "Age" 列
df$Age        # 访问 "Age" 列，返回向量 c(25, 30, 35)
df[["Age"]]   # 等价于 df$Age
df["Age"]     # 返回数据框(列名仍保留)
df[2, 3]      # 访问第2行第3列的值，输出 85
df[ , "Age"]  # 访问 "Age" 列，返回向量 c(25, 30, 35)
df[2, ]       # 访问第2行所有数据
df[, c(1, 3)] # 访问第1列和第3列
