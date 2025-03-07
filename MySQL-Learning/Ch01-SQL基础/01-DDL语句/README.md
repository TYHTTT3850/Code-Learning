# DDL 小结

## 数据库操作

```sql
show databases;

create database 数据库名;

use 数据库名;

select database();

drop database 数据库名
```

## 表操作

```sql
show tables;

create table 表名 (字段1 字段1类型,...,字段n 字段n类型);

desc 表名；

show create table 表名;

alter table 表名 add/modify/change/drop/rename to ... ;

drop table 表名;
```
