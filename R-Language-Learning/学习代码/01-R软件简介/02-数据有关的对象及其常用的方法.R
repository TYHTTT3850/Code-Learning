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

# ------------------------------------------------------------------------------------------------ #

### 因子：用于存储分类数据，存储的值是类别(levels)。

f <- factor(c("Male", "Female", "Male", "Female", "Male"))
levels(f)   #查看因子的类别
table(f)    #统计每个类别的频数

# ------------------------------------------------------------------------------------------------ #

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
solve(M)  # 返回逆矩阵，solve(A,b)则表示求解线性方程组 Ax=b

# 行/列求和 & 均值
rowSums(A)  # 计算每行的和
colSums(A)  # 计算每列的和
rowMeans(A) # 计算每行的均值
colMeans(A) # 计算每列的均值
# 行列式
det(A)  # 计算行列式

# ------------------------------------------------------------------------------------------------ #

### 列表：可以存储不同类型的数据对象。键值对的的形式存储，键=值。

lst <- list(name = "Alice", age = 25, scores = c(90, 85, 88))
lst$name       #访问元素，返回 "Alice"
lst[["name"]]   #返回 "Alice"
lst["name"]    #返回一个子列表，而不是字符串 "Alice"
length(lst)    #列表长度

# ------------------------------------------------------------------------------------------------ #

### 数据框：类似于表格，每列可以是不同的数据类型。每列都是向量，本质就是每列都是向量的列表，是列表的功能加强。

df <- data.frame(Name = c("A", "B", "C"), Age = c(25, 30, 35), Score = c(80, 90, 85))
dim(df)       #获取行列数
head(df)      #查看前6行
str(df)       #查看数据结构
summary(df)   #统计信息
df$Age        # 访问 "Age" 列，返回向量 c(25, 30, 35)
df[["Age"]]   # 等价于 df$Age
df["Age"]     # 返回数据框(列名仍保留)
df[2, 3]      # 访问第2行第3列的值，输出 85
df[ , "Age"]  # 访问 "Age" 列，返回向量 c(25, 30, 35)
df[2, ]       # 访问第2行所有数据
df[, c(1, 3)] # 访问第1列和第3列

# ------------------------------------------------------------------------------------------------ #

### 辨别和转换数据对象

# 创建各种类型的R对象用于演示
x_numeric <- 10.5
x_integer <- 10L
x_character <- "hello"
x_logical <- TRUE
x_complex <- 3+4i
x_vector <- c(1, 2, 3)
x_matrix <- matrix(1:9, nrow = 3)
x_dataframe <- data.frame(a = 1:3, b = c("x", "y", "z"))
x_list <- list(a = 1, b = "text", c = TRUE)
x_array <- array(1:24, dim = c(2, 3, 4))
x_factor <- factor(c("low", "medium", "high"))
x_null <- NULL
x_with_na <- c(1, 2, NA, 4)
x_with_nan <- c(1, 2, NaN, 4)

# 1. 检查对象类型的函数

# class() - 返回对象的类
class(x_numeric)      # [1] "numeric"
class(x_integer)      # [1] "integer"
class(x_character)    # [1] "character"
class(x_logical)      # [1] "logical"
class(x_complex)      # [1] "complex"
class(x_vector)       # [1] "numeric"
class(x_matrix)       # [1] "matrix" "array"
class(x_dataframe)    # [1] "data.frame"
class(x_list)         # [1] "list"
class(x_array)        # [1] "array"
class(x_factor)       # [1] "factor"
class(x_null)         # NULL

# typeof() - 返回对象的低级别类型
typeof(x_numeric)     # [1] "double"
typeof(x_integer)     # [1] "integer"
typeof(x_character)   # [1] "character"
typeof(x_logical)     # [1] "logical"
typeof(x_complex)     # [1] "complex"
typeof(x_matrix)      # [1] "integer"
typeof(x_dataframe)   # [1] "list"
typeof(x_list)        # [1] "list"

# mode() - 返回对象的模式
mode(x_numeric)       # [1] "numeric"
mode(x_integer)       # [1] "numeric"
mode(x_character)     # [1] "character"
mode(x_logical)       # [1] "logical"
mode(x_complex)       # [1] "complex"
mode(x_matrix)        # [1] "numeric"
mode(x_dataframe)     # [1] "list"
mode(x_list)          # [1] "list"

# 2.判断是否为某种对象

is.vector(x_vector)           # [1] TRUE
is.vector(x_matrix)           # [1] FALSE
is.matrix(x_matrix)           # [1] TRUE
is.matrix(x_vector)           # [1] FALSE
is.data.frame(x_dataframe)    # [1] TRUE
is.data.frame(x_matrix)       # [1] FALSE
is.list(x_list)               # [1] TRUE
is.list(x_dataframe)          # [1] TRUE (数据框也是列表的一种)
is.list(x_vector)             # [1] FALSE
is.array(x_array)             # [1] TRUE
is.array(x_matrix)            # [1] TRUE (矩阵也是数组的一种)
is.array(x_vector)            # [1] FALSE
is.numeric(x_numeric)         # [1] TRUE
is.numeric(x_integer)         # [1] TRUE
is.numeric(x_character)       # [1] FALSE
is.integer(x_integer)         # [1] TRUE
is.integer(x_numeric)         # [1] FALSE
is.double(x_numeric)          # [1] TRUE
is.double(x_integer)          # [1] FALSE
is.character(x_character)     # [1] TRUE
is.character(x_numeric)       # [1] FALSE
is.logical(x_logical)         # [1] TRUE
is.logical(x_numeric)         # [1] FALSE
is.factor(x_factor)           # [1] TRUE
is.factor(x_character)        # [1] FALSE
is.complex(x_complex)         # [1] TRUE
is.complex(x_numeric)         # [1] FALSE
is.null(x_null)               # [1] TRUE
is.null(x_numeric)            # [1] FALSE
is.na(x_with_na)              # [1] FALSE FALSE  TRUE FALSE
is.na(x_with_na[3])           # [1] TRUE
is.nan(x_with_nan)            # [1] FALSE FALSE  TRUE FALSE
is.nan(x_with_nan[3])         # [1] TRUE

# 3. 转换数据对象的类型，返回新的对象，不改变原对象

# as.vector() - 将对象转换为向量
as.vector(x_matrix)           # [1] 1 4 7 2 5 8 3 6 9 (按列展开矩阵)
as.vector(x_factor)           # [1] 1 2 3 (转换为因子的数字表示)

# as.matrix() - 将对象转换为矩阵
as.matrix(x_vector)           # 转为单列矩阵
as.matrix(x_dataframe)        # 将数据框转为矩阵

# as.data.frame() - 将对象转换为数据框
as.data.frame(x_matrix)       # 将矩阵转为数据框
as.data.frame(x_list)         # 将列表转为数据框

# as.list() - 将对象转换为列表
as.list(x_vector)             # 将向量转为列表
as.list(x_dataframe)          # 将数据框转为列表

# as.array() - 将对象转换为数组
as.array(x_vector)            # 将向量转为一维数组
as.array(x_matrix)            # 将矩阵转为二维数组

# as.numeric() - 将对象转换为数值型
as.numeric(x_character)       # 将字符转为数值，若字符非数字则返回NA
as.numeric(c("1", "2", "3"))  # [1] 1 2 3
as.numeric(x_logical)         # [1] 1 (TRUE转为1)
as.numeric(x_factor)          # 将因子转为数值 [1] 1 2 3

# as.integer() - 将对象转换为整数型
as.integer(x_numeric)         # [1] 10 (截断小数)
as.integer(x_logical)         # [1] 1
as.integer("42")              # [1] 42

# as.double() - 将对象转换为双精度型
as.double(x_integer)          # [1] 10
as.double(x_logical)          # [1] 1
as.double("10.5")             # [1] 10.5

# as.character() - 将对象转换为字符型
as.character(x_numeric)       # [1] "10.5"
as.character(x_logical)       # [1] "TRUE"
as.character(x_factor)        # [1] "low" "medium" "high"

# as.logical() - 将对象转换为逻辑型
as.logical(0)                 # [1] FALSE
as.logical(1)                 # [1] TRUE
as.logical(c(0, 1, 2))        # [1] FALSE TRUE TRUE (非0值都转为TRUE)
as.logical(c("TRUE", "FALSE")) # [1] TRUE FALSE

# as.factor() - 将对象转换为因子型
as.factor(x_character)        # 将字符转为因子
as.factor(c(1, 2, 2, 1))      # 将数值转为因子

# as.complex() - 将对象转换为复数型
as.complex(x_numeric)         # [1] 10.5+0i
as.complex(c(1, 2))           # [1] 1+0i 2+0i
as.complex("1+2i")            # [1] 1+2i

# as.null() - 总是返回NULL，不管输入什么
as.null(x_numeric)  # 返回 NULL，忽略输入
as.null(x_vector)   # 返回 NULL，忽略输入
as.null("anything") # 返回 NULL，忽略输入
