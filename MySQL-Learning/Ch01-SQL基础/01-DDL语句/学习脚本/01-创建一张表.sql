show databases; #查看有哪些数据库

create database if not exists test; #若不存在，则创建test数据库

use test; #使用test数据库 

select database(); #查看当前使用的数据库

create table tb_user( #创建一张表
id int comment '编号',
name varchar(50) comment '姓名',
age int comment '年龄',
gender varchar(1) comment '性别'
) comment '用户表';

desc tb_user; #查看表结构

show create table tb_user; #查看创建该表的具体语句
