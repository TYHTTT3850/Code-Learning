# R语言的分支和循环结构

# 1. if/else if/else 条件分支
# 基本语法：
# if (条件1) {
#   语句块1
# } 
# else if (条件2) {
#   语句块2
# } 
# else {
#   语句块3
# }

# 示例1：基本if/else
x <- 10
if (x > 5) {
  print("x大于5")
} else {
  print("x小于或等于5")
}
# 输出: [1] "x大于5"

# 示例2：if/else if/else多分支
score <- 85
if (score >= 90) {
  grade <- "A"
} else if (score >= 80) {
  grade <- "B"
} else if (score >= 70) {
  grade <- "C"
} else if (score >= 60) {
  grade <- "D"
} else {
  grade <- "F"
}
print(paste("成绩是:", grade))
# 输出: [1] "成绩是: B"

# 示例3：嵌套if
x <- 10
y <- 5
if (x > 0) {
  if (y > 0) {
    print("x和y都是正数")
  } else {
    print("x是正数，y不是正数")
  }
} else {
  print("x不是正数")
}
# 输出: [1] "x和y都是正数"

# ------------------------------------------------------------------------------------------------ #

# 2. switch语句
# 基本语法：
# switch(表达式,
#   值1 = 结果1,
#   值2 = 结果2,
#   ...
#   默认结果
# )

# 示例1：字符串匹配
fruit <- "apple"
result <- switch(fruit,
  apple = "这是一个苹果",
  banana = "这是一个香蕉",
  orange = "这是一个橙子",
  "未知水果"  # 默认值
)
print(result)
# 输出: [1] "这是一个苹果"

# 示例2：数值索引（不常用）
i <- 2
result <- switch(i,
  "第一项",
  "第二项",
  "第三项",
  "第四项"
)
print(result)
# 输出: [1] "第二项"

# 示例3：用于函数选择
operation <- "mean"
data <- c(1, 2, 3, 4, 5)
result <- switch(operation,
  mean = mean(data),
  median = median(data),
  sum = sum(data),
  max = max(data),
  min = min(data)
)
print(paste("计算结果是:", result))
# 输出: [1] "计算结果是: 3"

# ------------------------------------------------------------------------------------------------ #

# 3. for循环
# 基本语法：
# for (变量 in 序列) {
#   循环体
# }

# 示例1：遍历向量
fruits <- c("apple", "banana", "orange")
for (fruit in fruits) {
  print(paste("I like", fruit))
}
# 输出:
# [1] "I like apple"
# [1] "I like banana"
# [1] "I like orange"

# 示例2：使用索引
for (i in 1:5) {
  print(i^2)
}
# 输出:
# [1] 1
# [1] 4
# [1] 9
# [1] 16
# [1] 25

# 示例3：遍历列表
my_list <- list(a = 1:3, b = "hello", c = TRUE)
for (name in names(my_list)) {
  print(paste("List element", name, ":", my_list[[name]]))
}
# 由于输出较复杂，这里不展示

# 示例4：嵌套for循环（创建乘法表）
for (i in 1:5) {
  for (j in 1:5) {
    cat(i*j, "\t")
  }
  cat("\n")
}
# 输出：
# 1   2   3   4   5   
# 2   4   6   8   10  
# 3   6   9   12  15  
# 4   8   12  16  20  
# 5   10  15  20  25  

# ------------------------------------------------------------------------------------------------ #

# 4. while循环
# 基本语法：
# while (条件) {
#   循环体
# }

# 示例1：基本while循环
i <- 1
while (i <= 5) {
  print(i)
  i <- i + 1
}
# 输出:
# [1] 1
# [1] 2
# [1] 3
# [1] 4
# [1] 5

# 示例2：条件求和
sum <- 0
i <- 1
while (sum < 50) {
  sum <- sum + i
  i <- i + 1
}
print(paste("需要", i-1, "个数字使总和达到或超过50，最终总和是:", sum))
# 输出: [1] "需要 10 个数字使总和达到或超过50，最终总和是: 55"

# 示例3：带有break的while循环
i <- 1
while (TRUE) {  # 无限循环
  if (i > 5) {
    break  # 当i大于5时跳出循环
  }
  print(i)
  i <- i + 1
}
# 输出:
# [1] 1
# [1] 2
# [1] 3
# [1] 4
# [1] 5

# ------------------------------------------------------------------------------------------------ #

# 5. repeat循环
# 基本语法：
# repeat {
#   循环体
#   if (条件) break
# }

# 示例1：基本repeat循环
i <- 1
repeat {
  print(i)
  i <- i + 1
  if (i > 5) break
}
# 输出:
# [1] 1
# [1] 2
# [1] 3
# [1] 4
# [1] 5

# 示例2：随机数生成直到满足条件
set.seed(123)  # 设置随机种子以便结果可复现
repeat {
  x <- runif(1)  # 生成一个0到1之间的随机数
  print(x)
  if (x > 0.8) break
}
# 输出:
# [1] 0.2875775
# [1] 0.7883051
# [1] 0.4089769
# [1] 0.8830174

# 循环控制语句：break和next

# break示例：提前结束循环
for (i in 1:10) {
  if (i == 6) {
    break  # 当i等于6时结束循环
  }
  print(i)
}
# 输出:
# [1] 1
# [1] 2
# [1] 3
# [1] 4
# [1] 5

# next示例：跳过当前迭代
for (i in 1:10) {
  if (i %% 2 == 0) {
    next  # 跳过偶数
  }
  print(i)
}
# 输出:
# [1] 1
# [1] 3
# [1] 5
# [1] 7
# [1] 9
