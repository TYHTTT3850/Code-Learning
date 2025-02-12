# 序列：列表，元组，字符串。有序可重复
# 切片得到新序列，不改变原序列。序列名[起始索引:结束索引:步长]，不包括结束索引。
# 起始索引留空不写默认从头开始。
# 结束索引留空不写默认到尾结束。
# 步长位置留空不写默认为1。
list = [0,1,2,3,4,5,6,7,8,9]
tuple = (0,1,2,3,4,5,6,7,8,9)
string = "hello world"
# 正序切片，从左往右,索引从0(最左边)开始，步长 > 0
list_cut1 = list[2:4]
list_cut2 = list[:4]
list_cut3 = list[2:]
list_cut4 = list[::2]
list_cut5 = list[::1]
print(list_cut1)
print(list_cut2)
print(list_cut3)
print(list_cut4)
print(list_cut5)
# 逆序切片，从右往左,索引从-1(最右边)开始,步长 < 0
list_cut6 = list[-2:-4:-1]
list_cut7 = list[:-4:-1]
list_cut8 = list[-2::-1]
list_cut9 = list[::-2]
list_cut10 = list[::-1]
print(list_cut6)
print(list_cut7)
print(list_cut8)
print(list_cut9)
print(list_cut10)
# 元组和字符串也一样
