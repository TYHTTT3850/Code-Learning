# datetime标准库提供了处理日期和时间的类和函数

import datetime

"""1、datetime.datetime 类，日期 + 时间（最常用）"""
#创建datetime对象
dt = datetime.datetime(2025,5,1,14,30,45)
print(dt)

#获取本地当前时间
now = datetime.datetime.now()
print(now)

#获取UTC当前时间
utcnow = datetime.datetime.now(datetime.UTC)
print(utcnow)

#格式化与解析
dt = datetime.datetime(2025,5,1,14,30,45)

#日期转字符串
date_str = dt.strftime('%Y-%m-%d-%H-%M-%S')
print(date_str,type(date_str))

#字符串转换日期
dt = dt.strptime("2025-05-01 23:30:45", "%Y-%m-%d %H:%M:%S")
print(dt,type(dt))

#属性访问
year = dt.year    # 年
month = dt.month  # 月
day = dt.day      # 日
hour = dt.hour    # 时
minute = dt.minute  # 分
second = dt.second  # 秒
microsecond = dt.microsecond  # 微秒
print(year,month,day,hour,minute,second,microsecond)

# 替换日期时间的某部分
new_dt = dt.replace(year=2026, month=6)
print(new_dt)

# 获取星期几 (0是周一，6是周日)
weekday = dt.weekday()
print(weekday)
weekday = dt.isoweekday() #1是周一，7是周日
print(weekday)

# 获取时间戳
#时间戳是指格林威治时间1970年01月01日00时00分00秒(北京时间1970年01月01日08时00分00秒)起至现在的总秒数
timesamp = dt.timestamp()
print(timesamp)
"""2、datetime.date类，只包含日期，其他方法与 datetime.datetime 类似"""

"""3、datetime.time类，只包含时间，其他方法与 datetime.datetime 类似"""

"""4、datetime.timedelta()方法"""
tomorrow = dt + datetime.timedelta(days=1)
yesterday = dt - datetime.timedelta(days=1)
next_hour = dt + datetime.timedelta(hours=1)
diff = tomorrow - yesterday  # 返回timedelta对象
print(tomorrow,yesterday,next_hour,diff,sep='\n')