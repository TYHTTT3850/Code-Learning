"""
当数据容器中不止一种数据类型时，需要使用 Union 联合类型注解。
除了 Union 联合类型注解外，还可以这样进行注解：类型1|类型2|...|类型N 。
两种注解都适用于变量和函数(方法)且两者等价。
"""

# Union[类型1,类型2,...,类型N]
from typing import Union
list1:list[Union[int,str,float]] = [1,"abc",3.1415926]
dict1:dict[str,Union[int,str]]={"a":1,"b":"qqq"}
def func1(data:Union[int,str]) -> Union[int,str]:
    pass

# 类型1|类型2|...|类型N
list2:list[int|str] = [1,"abc"]
dict2:dict[str,int|str]={"a":2,"b":"www"}
def func2(data:int|float) -> int|float:
    pass