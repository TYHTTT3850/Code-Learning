"""
展示的全是定义在序列(Series)类和数据框(DataFrame)类中的方法
"""

import pandas as pd
s = pd.Series([None,32,18,27,19,26,24],
              index=['A','B','C','D','E','F','G']
              )
df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie',"Doss","Eve","Fog","Gross"],
                   'age': [None, 32, 18,27,19,26,24]}
                  )

"""--------------------查看数据--------------------"""
print("--------------------查看数据--------------------")

# head()方法。查看前几行数据
print(s.head(5),end="\n--------------\n") #前五行
print(df.head(5),end="\n--------------\n")

# tail()方法。查看后几行数据
print(s.tail(5),end="\n--------------\n")
print(df.tail(5),end="\n--------------\n")

# describe()方法。查看数据的描述性统计信息
print(s.describe(),end="\n--------------\n")
print(df.describe(),end="\n--------------\n")

"""--------------------操作数据--------------------"""
print("--------------------操作数据--------------------")

# drop()方法。删除某些行或列
s1 = s.drop('A')
print(s1,end="\n--------------\n")

df1 = df.drop("name",axis=1)
print(df1,end="\n--------------\n")

# merge()方法。类似SQL的JOIN操作
df_left = pd.DataFrame({'id':[1,2,3], 'val1':['a','b','c']})
df_right = pd.DataFrame({'id':[2,3,4], 'val2':['x','y','z']})
df_merge = pd.merge(df_left, df_right, on="id", how="inner")
print(df_merge,end="\n--------------\n")

# concat()方法。直接拼接
df_top = pd.DataFrame({'col1':[1,2], 'col2':[3,4]})
df_bottom = pd.DataFrame({'col1':[5,6], 'col2':[7,8]})
df_concat = pd.concat([df_top, df_bottom], axis=0, ignore_index=True)
print(df_concat,end="\n--------------\n")

# join()方法。默认用索引对齐
df_left = pd.DataFrame({'A':[1,2,3]}, index=['x','y','z'])
df_right = pd.DataFrame({'B':[4,5,6]}, index=['x','y','w'])
df_join = df_left.join(df_right, how="outer")
print(df_join,end="\n--------------\n")

"""
键：某一列(如id这一列)
键值：某一列上的值(如id这一列上具体的值1、2、3等)
how 参数效果对比：
 ┌─────────┬─────────────────────┬────────────────────────────┐
 │  how    │   保留哪些键值     │          效果说明         │
 ├─────────┼─────────────────────┼────────────────────────────┤
 │ inner   │ 两表键值的交集     │ 只保留两边都有的键值     │
 │ left    │ 左表全部键值       │ 左表为基准,右表缺失填NaN │
 │ right   │ 右表全部键值       │ 右表为基准,左表缺失填NaN │
 │ outer   │ 两表键值的并集     │ 保留所有键值,缺失处填NaN │
 └─────────┴─────────────────────┴────────────────────────────┘
"""

"""--------------------处理缺失值--------------------"""
print("--------------------处理缺失值--------------------")

# isnull()方法。检查哪些位置是缺失值
print(s.isnull(),end="\n--------------\n")
print(df.isnull(),end="\n--------------\n")

# fillna()方法。填充缺失值
s.fillna(25,inplace=True) #inplace表示是否替换原序列，要不然返回新序列
print(s,end="\n--------------\n")

df_filled = df.fillna(25)
print(df_filled,end="\n--------------\n")

"""--------------------排序和排名--------------------"""
print("--------------------排序和排名--------------------")

# sort_values()方法。按照某一列的值进行升序或降序排列
s2 = s.sort_values(ascending=True)
print(s2,end="\n--------------\n")

df2 = df.sort_values("age",ascending=False)
print(df2,end="\n--------------\n")

# rank()方法。对数据排名，返回排名结果
rank = s.rank(ascending=False)
print(rank,end="\n--------------\n")

df.loc[:,"rank"] = df.loc[:,"age"].rank(method="dense",ascending=False) #对"age"列排名

print(df,end="\n--------------\n")

