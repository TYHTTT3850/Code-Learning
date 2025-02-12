# 集合无序(无索引)且不重复，可更改
set1 = {0,1,2,3,4,5,6,7,8,9}
set_empty = set()

# add方法，添加指定元素。集合名.add(指定元素)
set1.add(10)
print(set1)

# remove方法，删除指定元素。集合名.remove(指定元素)
set1.remove(0)
print(set1)

# pop方法，删除一个随机的元素，返回删除元素值。集合名.pop()
set3 = {"a","b","c"}
pop_element = set3.pop()
print(pop_element)
print(set3)

# clear方法，清空集合。集合名.clear()
set1.clear()
print(set1)

# difference方法，求集合1和集合2的差，不改变原集合，返回新集合。集合名1.difference(集合名2)=集合名1-集合名2
set4 = {1,2,3}
set5 = {1,5,6}
print(set4.difference(set5),set4-set5) # set4有而set5没有
print(set5.difference(set4),set5-set4) # set5有而set4没有
print(set4,set5)

# difference_updata方法，求集合1与集合2的差，改变集合1，不改变集合2。集合名1.difference_updata(集合名2)
set4.difference_update(set5)
print(set4)
print(set5)

# union方法，求两个集合的并，得到新集合，不改变原集合。集合名1.union(集合名2)=集合名1|集合名2
set6 = {1,2,3}
set7 = {1,5,6}
set8 = set6.union(set7)
print(set6,set7,set8,set6|set7)

# intersection方法，求两个集合的交，得到新集合，不改变原集合。集合名1.intersection(集合名2)=集合名1&集合名2
set9 = set6.intersection(set7)
print(set6,set7,set9,set6&set7)

# symmetric_difference方法，求两个集合的对称差，得到新集合，不改变原集合。集合名1.symmetric_difference(集合名2)=集合名1^集合名2
set10 = set6.symmetric_difference(set7)
print(set6,set7,set10,set6^set7)

# 统计集合元素个数。len(集合名)
set11 = {1,2,3,4,5}
set12 = {1,2,3,4,5,1,2,3,4,5} #会去重
print(len(set11))
print(len(set12))
