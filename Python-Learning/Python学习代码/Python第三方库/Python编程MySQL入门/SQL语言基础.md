# SQL语言特点

1、SQL语言大小写不敏感。例如：

```sql
show databases;
```

```sql
SHOW Databases;
```

2、SQL语言可以多行书写，但最后必须以分号结尾。例如：

```sql
show databases;
```

```sql
show
databases;
```

3、SQL支持注释，也有单行和多行注释：

单行注释(注意--后有空格，否则报错。#后可以不要空格，但建议加上)：

```sql
-- 单行注释
```

```sql
# 单行注释
```

多行注释：

```sql
/*
多
行
注
释
*/
```

**SQL中出现字符串的地方，都必须用单引号包围(字符串化)，否则报错。**

**[]中的内容表示可选参数，在写代码时只需将[]内的内容填入代码中即可，不需要加上[]。**

# SQL语言分类

## 数据定义DDL(Data Definition Language)：

  库的创建删除，表的创建删除等。

### 库管理

1、查看有哪些数据库：

```sql
show databases;
```

2、使用数据库：

```sql
use 数据库名;
```

3、创建数据库：

```sql
creat database 数据库名 [charset utf8];# []内表示可选参数，表示字符集编码为utf8
```

4、删除数据库：

```sql
drop database 数据库名;
```

5、查看当前使用的数据库：

```sql
select database();
```

### 表管理(要先选择数据库)：

1、查看有哪些表：

```sql
show tables;
```

2、创建表：

```sql
create table 表名称(
    列名称1 列类型,
    列名称2 列类型,
    ……
    列名称N 列类型
);
```

入门列类型：

```sql
int            -- 整数
float          -- 浮点数
varchar(长度)   -- 字符串，长度填入数字，做字符串长度限制。长度可填入的最大值为255
date           -- 日期类型
timestamp      -- 时间戳类型
```

3、删除表

```sql
drop table 表名;
```

## 数据操纵DML(Data Manipulation Language)：

新增数据，删除数据，修改数据等。

### 插入数据

基础语法：

```sql
insert into 表名[(列1,列2,……,列N)] values(值1,值2,……,值N)[,(值1,值2,……,值N),……,(值1,值2,……,值N)]; #列1——列N 与 值1——值N 一一对应
```

例如：

```sql
insert into student(id) values (1),(2),(3); #一列插入多行值
```

```sql
insert into student (id,name,age) values (4,'Jack',31),(5,'Mike',31); 
#字符串插入时必须用单引号围起来(字符串化)，否则报错
```

### 删除数据

基础语法：

```sql
delete from 表名 [where 条件判断]; #条件判断：列 操作符 值
#操作符：=、<、>、>=、<=、!= 等
```

例如：

```sql
delete from student where id >=2; #不加where会导致表内全部数据删除
```

### 更新数据

基础语法：

```sql
update 表名 set 列=值 [where 条件判断]; 
```

例如：

```sql
update student set name = 'Jack' where id = 1; #不加where会导致整个name列变为Jack
```

## 数据控制DCL(Data Control Language)：

新增用户，删除用户，密码修改，权限管理等。

## 数据查询DQL(Data Query Language)：

基于需求查询和计算数据等。

### 基础查询

基础语法：

```sql
select 列名1,……,列名N from 表名 [where 条件判断];
```

```sql
select * from 表名 [where 条件判断];# *号表示全部的列
```

例如：

```sql
select id,name from student;
```

```sql
select * from student;
```

```sql
select id,name from student where id >=4
```

```sql
select id,name from student where name='jack';
```

### 分组(GROUP BY)聚合(aggregate)

基础语法：

```sql
#group by中出现了哪一列，select中才能出现哪一列，聚合函数内除外。
select 列|聚合函数 from 表 [where 条件判断] group by 列;
#聚合函数有：
-- sum(列)       求和
-- avg(列)       求平均值
-- min(列)       求最小值
-- max(列)       求最大值
-- count(列|*) 求数量
```

例如：

```sql
select gender,sum(age),avg(age),min(age),max(age),count(*) from student group by gender;
```

### 排序(ORDER BY)

基础语法：

```sql
select 列|聚合函数 from 表 [where 条件判断] group by 列 order by 列 [asc|desc];
-- asc 升序，默认
-- desc 降序
```

例如：

```sql
select * from student order by age;
```

### 分页(LIMIT)

```sql
select 列|聚合函数 from 表 [where 条件判断] group by 列 order by 列 [asc|desc] limit n[,m];
```

例如：

```sql
select * from student limit 5; #从头开始取5条数据
```

```sql
select * from student limit 2,3; #从第二条往后(不包括第二条)取三条数据
```

### 注意事项

1、关键字顺序不能调换，否则报错。例如group by 写到 where 前就是错的。

2、一个查询语句中，select 和 from 关键字是必须的，其他都是可选的。

3、语句执行顺序：from --> where --> group by --> 聚合函数 --> select --> order by --> limit
