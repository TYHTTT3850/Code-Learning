# 导入模块
from pymysql import Connection

# 创建连接对象
conn = Connection(
    host="localhost",
    port=3306,
    user="root",
    passwd="Tyh11811.",
    # 设置自动确认提交
    # autocommit=True,
)
print(conn.get_server_info()) # 查看连接信息

# 创建游标对象
cursor = conn.cursor()

# 在pymysql中输入SQL语句时不需要在结尾加上;号

# 创建数据库
cursor.execute("create database pymysql_test")

# 选择数据库
conn.select_db("pymysql_test")

# 创建表
cursor.execute("create table student (id int,name varchar(10),age int,gender varchar(2))")

# 执行SQL插入语句
cursor.execute("insert into student values(10001,'Mike',31,'男'),(10002,'Jack',31,'男')")

# 确认提交，否则不会对表中的数据进行更改
conn.commit()

# 执行SQL查询语句
cursor.execute("select * from student")
results = cursor.fetchall() # 将得到的查询结果封装到元组内
print(results)

# 关闭连接
conn.close()
