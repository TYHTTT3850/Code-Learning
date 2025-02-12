# 字符串不可更改
string1 = "hello world"

# 字符串索引
s1 = string1[0]
s2 = string1[-5]
print(s1,s2)

# index方法，查找某字符串中第一个字第一次出现的索引。字符串名.index(字符串)
print(string1.index("world"))

# replace方法，将字符串1的内容替换为字符串2并返回新的字符串。字符串名.replace(字符串1，字符串2)
new_string1 = string1.replace("hello","HELLO")
print(f"{string1}变为{new_string1}")

# split方法，按照指定分隔符分割字符串，返回一个字符串列表。字符串名.split("分隔符")
string2 = "HELLO WORLD hello world"
string2_list = string2.split(" ") #空格为分隔符
print(f"{string2}以空格为分隔符分割后得到{string2_list}")

# strip方法，去除字符串首尾指定字符，默认删除空白字符。字符串名.strip(指定字符)
string3 = "   hello world  "
string4 = "123hello123 321world321"
print(string3.strip()) #默认删除空格
print(string4.strip("123")) #删除1，2，3
"""
空白字符：空格(' ')，制表符('\t')，换行符('\n')，回车符('\r')，垂直制表符('\v')，换页符('\f')
"""

# count方法，统计指定字符串出现的次数。字符串名.count(指定字符串)
print(string1.count('l'))
print(string1.count("ll"))

#统计字符串长度。len(字符串名)
print(len(string1))