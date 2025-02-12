# JSON是完全独立于编程语言，采用文本(字符串)形式来存储数据的轻量级数据交互格式
# JSON数据格式和Python数据格式可无缝转换
# JSON主要用于不同编程语言和程序间的数据交换，即中转数据格式
import json
# Python数据转换为JSON数据
data1 = [{"name":"王大锤","age":13},{"name":"张大山","age":11},{"name":"赵小虎","age":16}]
json_data11 = json.dumps(data1)
print(type(json_data11), json_data11,sep=" ; ") # json_data数据类型为字符串
json_data12 = json.dumps(data1,ensure_ascii=False) # 正常展示中文
print(json_data12)
data2 = {"name":"周杰轮","address":"台北"}
json_data21 = json.dumps(data2,ensure_ascii=False)
print(json_data21)
print()

# JSON数据转换为Python数据
data_json1 = '[{"name": "王大锤", "age": 13}, {"name": "张大山", "age": 11}, {"name": "赵小虎", "age": 16}]'
Python_data1 = json.loads(data_json1)
print(type(Python_data1),Python_data1,sep=" ; ")
data_json2 = '{"name":"周杰轮","address":"台北"}'
Python_data2 = json.loads(data_json2)
print(Python_data2)
