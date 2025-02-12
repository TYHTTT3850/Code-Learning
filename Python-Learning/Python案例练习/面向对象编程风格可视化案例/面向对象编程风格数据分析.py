"""
1.设计一个类，可以完成数据封装。
2.设计一个抽象类，定义文件读取的相关功能，并使用子类实现具体功能。
3.读取文件，生产数据对象。
4.计算每一天的销售额。
5.通过Pyecharts实现可视化
"""
import json
from pyecharts.charts import Bar
from pyecharts.options import *
from pyecharts.globals import ThemeType

#步骤1
class Record:
    def __init__(self,date:str,order_id:str,money:int,province:str):
        self.date = date
        self.order_id = order_id
        self.money = money
        self.province = province

    def __str__(self):
        return f"{self.date},{self.order_id},{self.money},{self.province}"


#步骤2
class FileReader:
    def __init__(self,path):
        self.path = path
    def read(self):
        pass

#步骤3：设计txt文件读取方法
class TextFileReader(FileReader):
    def read(self)->list[Record]:
        f = open(self.path,"r",encoding="utf-8")
        record_list:list[Record] = []
        for line in f.readlines():
            line = line.strip("\n") #去除结尾的换行符
            line_list = line.split(",")
            record = Record(line_list[0],line_list[1],int(line_list[2]),line_list[3])
            record_list.append(record)
        f.close()
        return record_list

#步骤3：设计json文件读取方法
class JsonFileReader(FileReader):
    def read(self)->list[Record]:
        f = open(self.path,"r",encoding="utf-8")
        record_list:list[Record] = []
        for line in f.readlines():
            line = line.strip("\n") #去除结尾的换行符
            line_dict = json.loads(line)
            record = Record(line_dict["date"],line_dict["order_id"],line_dict["money"],line_dict["province"])
            record_list.append(record)
        f.close()
        return record_list

#步骤3：读取数据
text_file_reader = TextFileReader("2011年1月销售数据.txt")
json_file_reader = JsonFileReader("2011年2月销售数据JSON.txt")
Jan_data:list[Record] = text_file_reader.read()
Feb_data:list[Record] = json_file_reader.read()
all_data:list[Record] = Jan_data + Feb_data

#步骤4：将不同地区的同日销售数据求和
everyday_data_dict:dict = {}
for record in all_data:
    if record.date in everyday_data_dict:
        everyday_data_dict[record.date] += record.money
    else:
        everyday_data_dict[record.date] = record.money

#步骤4：将字典的键和值分别放入两个列表中，以便画图
bar = Bar({"theme":ThemeType.LIGHT})
bar.add_xaxis(list(everyday_data_dict.keys()))
bar.add_yaxis(series_name="每日销售额",y_axis=list(everyday_data_dict.values()),label_opts=LabelOpts(is_show=False))
bar.set_global_opts(
    title_opts=TitleOpts(title="1月-2月每日销售额柱状图",pos_left="center",pos_bottom="1%"),#标题配置
)

bar.render("每日销售额柱状图.html")
