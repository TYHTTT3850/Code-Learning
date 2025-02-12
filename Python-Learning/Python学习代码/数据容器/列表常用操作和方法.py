# 列表可更改
list1 = [1,2,3,4,5,6,7,8,9]

# index方法，查找某元素的索引。列表名.index(指定元素)
index = list1.index(7)
print(f"索引为：{index}")

# 修改元素。列表名[索引] = 新的值
list1[4] = 10
print(f"list1列表修改为：{list1}")

# insert方法，插入新的元素。列表名.insert(索引，元素)
list1.insert(3,11)
print(f"list1列表修改为：{list1}")

# append方法，将新元素追加到列表尾部。列表名.append(新元素)
list1.append(12)
print(f"list1列表修改为：{list1}")

# extend方法，追加一批元素到列表尾部，即合并两个列表。列表名.extend(其他数据容器)
list1.extend([13,14])
print(f"list1列表修改为：{list1}")

# 删除元素。del 列表[索引]
del list1[5]
print(f"list1列表修改为：{list1}")

# pop方法，删除对应索引位置元素，返回删除元素的值。列表名.pop(索引)
pop_element = list1.pop(11)
print(f"取出元素为{pop_element}")
print(f"list1列表修改为：{list1}")

# remove方法，删除指定元素(列表中的第一个匹配项)。列表名.remove(指定元素)
list1.remove(13)
print(f"list1列表修改为：{list1}")

# clear方法，清空列表。列表名.clear()
list1.clear()
print(f"list1列表修改为：{list1}")

# count方法，统计某个元素出现的次数。列表名.count(指定元素)
list2 = [1,1,2,2,2,3,3,4,4,4,4,5,5,5]
print(list2.count(2))

# 统计列表全部元素数量。len(列表名)
print(len(list2))