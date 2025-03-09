### 向量：R语言的基本数据结构，由相同类型的元素组成。

v <- c(10, 20, 30, 40, 50)  #创建数值型向量
v2 <- c("A", "B", "C") #字符型向量
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
dim(m)       #查看维度
t(m)         #转置
rowSums(m)   #计算每行的总和
colMeans(m)  #计算每列的均值
rownames(m)  #获取行名
colnames(m)  #获取列名
rownames(m) <- NULL  # 删除行名
colnames(m) <- NULL  # 删除列名


### 列表：可以存储不同类型的数据对象。键值对的的形式存储，键=值。

lst <- list(name = "Alice", age = 25, scores = c(90, 85, 88))
lst$name       #访问元素，返回 "Alice"
lst[["name"]]   #返回 "Alice"
lst["name"]    #返回一个子列表，而不是字符串 "Alice"
length(lst)    #列表长度


### 数据框：类似于表格，每列可以是不同的数据类型。每列都是列表

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
