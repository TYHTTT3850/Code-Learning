# 数据容器：列表，元组，字符串，集合，字典。
list1 = [1,2,3,4,5,6]
tuple1 = (1,2,3,4,5,6)
string1 = "abcdefg"
set1 = {1,2,3,4,5,6}
dictionary1 = {"key1":11,"key2":22,"key3":33,"key4":44,"key5":55,"key6":66}

# 统计元素个数。len(容器)
print(f"列表元素个数：{len(list1)}")
print(f"元组元素个数：{len(tuple1)}")
print(f"字符串元素个数：{len(string1)}")
print(f"集合元素个数：{len(set1)}")
print(f"字典元素个数：{len(dictionary1)}")
print()

#求最大值。max(容器)
print(f"列表元素最大值{max(list1)}")
print(f"元组元素最大值{max(tuple1)}")
print(f"字符串元素最大值{max(string1)}")
print(f"集合元素最大值{max(set1)}")
print(f"字典元素最大值{max(dictionary1)}")
print()

#求最小值。min(容器)
print(f"列表元素最小值{min(list1)}")
print(f"元组元素最小值{min(tuple1)}")
print(f"字符串元素最小值{min(string1)}")
print(f"集合元素最小值{min(set1)}")
print(f"字典元素最小值{min(dictionary1)}")
print()

# 类型转换为列表。list(容器)
print(f"元组转换为列表{list(tuple1)}")
print(f"字符串转换为列表{list(string1)}")
print(f"集合转换为列表{list(set1)}")
print(f"字典转换为列表{list(dictionary1)}")
print()

# 类型转换为元组。tuple(容器)
print(f"列表转换为元组{tuple(list1)}")
print(f"字符串转换为元组{tuple(string1)}")
print(f"集合转换为元组{tuple(set1)}")
print(f"字典转换为元组{tuple(dictionary1)}")
print()

# 类型转换为字符串。str(容器)
print(f"列表转换为字符串{str(list1)}") # 转换为"[1,2,3,4,5,6]"，输出时引号隐藏，下同。
print(f"元组转换为字符串{str(tuple1)}")
print(f"集合转换为字符串{str(set1)}")
print(f"字典转换为字符串{str(dictionary1)}")
print()

# 类型转换为集合。set(容器)
print(f"列表转换为集合{set(list1)}") # 转换为"[1,2,3,4,5,6]"，输出时引号隐藏，下同。
print(f"元组转换为集合{set(tuple1)}")
print(f"字符串转换为集合{set(string1)}")
print(f"字典转换为集合{set(dictionary1)}")
print()
# 列表，元组，字符串，集合无法转为字典，因为缺少键值对。

# 排序操作。sorted(容器)，对容器内元素排序并返回排序好的列表。
list2 = [3,1,4,5,2,9]
tuple2 = (3,1,4,5,2,9)
string2 = "abcdefg"
set2 = {3,1,4,5,2,9}
dictionary2 = {"key3":11,"key1":22,"key4":33,"key5":44,"key2":55,"key9":66}
list2_sorted =  sorted(list2)
tuple2_sorted =  sorted(tuple2)
string2_sorted =  sorted(string2)
set2_sorted =  sorted(set2)
dictionary2_sorted =  sorted(dictionary2)
print(list2_sorted, tuple2_sorted, string2_sorted, set2_sorted, dictionary2_sorted,sep="\n")