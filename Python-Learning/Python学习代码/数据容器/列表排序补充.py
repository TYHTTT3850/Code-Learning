list1 = [1,2,3,4,5,6,7,8,9]

# 列表排序。sorted(列表名)，返回一个新列表，不改变原列表
list1_sorted = sorted(list1, reverse=True) # 逆序排列
print(list1,list1_sorted,sep=" ")

"""
sort方法,会改变原列表。列表名.sort(key=选择依据排序的函数名(注意是函数名！),reverse=True or False)
key：将每个列表元素传入该函数得到返回值后根据函数返回值排序
"""
# 无排序依据的sort方法
list1.sort(reverse=True)
print(list1)

# 有排序依据的sort方法
def Choose_Sort_Key(element):
    return element[1]
list2 = [["a",11],["b",12],["c",13],["d",14]]
list2.sort(key=Choose_Sort_Key,reverse=True)
print(list2)

# lambda函数作为key
list3 = [["a",11],["b",13],["c",12]]
list3.sort(key=lambda element:element[1])
print(list3)