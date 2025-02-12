# 元组不可更改
tuple1 = (1,2,3,4,5,6,7,8,9)
tuple_single = (1,) #single element tuple。加逗号表示其为元组，否则数据类型不为元组。

# index方法，查找某元素的索引。元组名.index(指定元素)
index = tuple1.index(3)
print(f"索引为：{index}")

# count方法，统计某个元素出现的次数。元组名.count(指定元素)
tuple2 = (1,1,1,2,2,2,2,3,3,4,5,6,7,8,9)
count = tuple2.count(2)
print(count)

# 统计元组全部元素数量。len(元组名)
print(len(tuple1))