dictionary1 = {"key1":1,"key2":2,"key3":3}
dictionary2 = {"键名1":1,"键名2":2,"键名2":3} #有重复键名的字典，后面的值覆盖前面的
dictionary3 = {1:'a',2:'b',3:'c'}
dictionary_empty1 = {} #空字典1
dictionary_empty2 = dict() #空字典2
print(dictionary1,dictionary2,dictionary3,sep=" ; ")

# 字典索引。字典名[键名]
print(dictionary1["key1"],dictionary2["键名2"],dictionary3[1],sep=" ; ")

# 增加元素。字典名[不存在的键名]
dictionary1["key4"] = 4
print(dictionary1)

# 修改元素。字典名[存在的键名] = 新值
dictionary1["key1"] = 0
print(dictionary1)

# pop方法，删除对应键值的元素，返回删除元素的值。字典名.pop(键值)
pop_value = dictionary1.pop("key1")
print(dictionary1,pop_value,sep = " ; ")

# clear方法，清空字典。字典名.clear()
dictionary1.clear()
print(dictionary1)

# keys方法，获取字典全部键名。字典名.keys()
dictionary3_keys = dictionary3.keys()
print(dictionary3_keys)

#values方法，获取字典全部值。字典名.values()
dictionary3_values = dictionary3.values()
print(dictionary3_values)

# 统计字典元素数量。len(字典名)
length = len(dictionary2)
print(length)