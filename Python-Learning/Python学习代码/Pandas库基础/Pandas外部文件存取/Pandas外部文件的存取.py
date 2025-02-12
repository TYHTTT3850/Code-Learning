"""
实际应用中，通常通过 Python 读取外部文件: 1、文本文件(csv，txt)。
                                    2、电子表格Excel文件。
"""
import pandas as pd
# pd.set_option("display.float_format","{:.0f}".format) 禁用科学计数法

"""--------------------文本文件读取--------------------"""
print("--------------------文本文件读取--------------------")
# read_csv()方法，可以读取txt或csv文本格式数据
a = pd.read_csv("Pandas库txt读取示例.txt", sep=",",
                header=None) #sep:数据分隔符，header:列名，默认把第一行(下标为0)当作列名,即默认header=0
print(a,end="\n--------------\n")

b = pd.read_csv("Pandas库txt读取示例.txt", sep=",",
                names=['fly','game','ice_cream'])#自定义列名，可以省略header=None
print(b,end="\n--------------\n")

c = pd.read_csv("Pandas库csv读取示例.csv", sep=",",
                dtype={"GDP":int}, index_col="country",
                usecols=["country","GDP"],
                skiprows=lambda x: x > 0 and x % 2 != 0,
                skipfooter=3,
                engine="python")
#对dtype参数进行设置以防止显示科学计数法
#index_col:指定某列或某些列为索引(指定多列需放在列表里)
#usecols:指定读取哪些列
#engine:pandas解析数据时用的引擎。默认为c，c引擎解析速度快，但是特性没有python引擎全
#skiprows:过滤掉指定的行，可以传入函数，如skiprows=lambda x: x > 0 and x % 2 != 0
#skipfooter:从文件末尾过滤行，解析引擎退化为 Python。这是因为 C 解析引擎没有这个特性。
#skipfooter:从结尾往上过滤掉指定数量的行，要手动指定engine="python"
print(c,end="\n--------------\n")
print(c.loc["中国"]) #使用loc索引器使用index_col指定的索引

d = pd.read_csv("Pandas库csv读取示例.csv", sep=",", nrows=10)
print(d,end="\n--------------\n")
#nrows:设置一次性读入的文件行数，在读入大文件时很有用,不能和skipfooters一起用
print()

"""--------------------Excel文件的存取--------------------"""
print("--------------------Excel文件的存取--------------------")

# 读取数据
a = pd.read_excel("Pandas库读取Excel文件示例.xlsx",
                  usecols=range(1,4),#提取第2-4列数据
                  )
print(a,end="\n--------------\n")
print(a.describe(),end="\n--------------\n")

# 写入数据
b = a.values
print(b,end="\n--------------\n")
c = pd.DataFrame(b,index=range(1,11),columns=["用户A","用户B","用户C"])
f = pd.ExcelWriter("Pandas库写入Excel文件示例.xlsx") #创建文件对象
c.to_excel(f,sheet_name="sheet1") #写入表单1
c.to_excel(f,sheet_name="sheet2") #写入表单2
f._save()