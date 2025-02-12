"""
正则表达式：使用单个字符串来匹配某个句法规则的字符串，常用来检索替换符合某个模式(规则)的字符串
"""

import re

"""re.match()方法。从开头匹配字符串，若开头都不匹配，后面的内容都不管。re.match(匹配规则,被匹配字符串)"""

s = "python it python it python it"
result = re.match("python", s)
print(result) #<re.Match object; span=(0, 6), match='python'>
print(result.span()) #(0, 6)，左闭右开区间
print(result.group()) #python

s = "1python it python it python it"
result = re.match("python", s)
print(result) #None

"""re.search()方法。从开头匹配字符串，找到第一个匹配的就停止，没找到就返回None。re.search(匹配规则,被匹配字符串)"""

s = "1python it python it python it"
result = re.search("python", s)
print(result) #<re.Match object; span=(1, 7), match='python'>

"""re.findall()方法。找出所有匹配的字符串，没找到就返回空list。re.findall(匹配规则,被匹配字符串)"""

s = "1python it python it python it"
result = re.findall("python", s)
print(result) #['python', 'python', 'python']