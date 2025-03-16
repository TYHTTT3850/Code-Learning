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

# 也可以用来读写 .data 文件。

### 读写Excel文件

# 读
library(readxl)
data <- read_excel("data.xlsx", sheet = 1) #sheet=1表示读取第一页数据，也可以按照名称的方式的读取，如：sheet = "Sheet2"

# 写
library(writexl)
write_xlsx(data, "output.xlsx")


### 读写JSON

# 读
library(jsonlite)
data <- fromJSON("data.json")

# 写
write_json(data, "output.json")


### 读写R数据格式(RDS & RData)

# 读
data <- readRDS("data.rds")
load("data.RData") #会直接加载数据文件中的所有对象，不需要显式赋值。

# 写
saveRDS(data, "output.rds")
save(data, file = "output.RData")