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