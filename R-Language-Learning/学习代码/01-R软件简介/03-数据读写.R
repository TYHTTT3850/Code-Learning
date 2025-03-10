### 读写CSV文件

# 读
data <- read.csv("data.csv", header = TRUE, stringsAsFactors = FALSE) #header=TRUE表示第一行为表头，stringsAsFactors = FALSE表示不要将字符串转化为因子

# 写
write.csv(data, "output.csv", row.names = FALSE) #row.names = FALSE：不写入行名，文件中不包含额外的行名列。

### 读写TXT文件

#读
data <- read.table("data.txt", header = TRUE, sep = "\t", stringsAsFactors = FALSE)

#写
write.table(data, "output.txt", sep = "\t", row.names = FALSE, quote = FALSE)
