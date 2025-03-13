create table employee(id int , workno int, name varchar(10), gender char(1), age tinyint unsigned,
                      idcard char(18), workdate date);

alter table employee add nickname varchar(20) comment '昵称'; # 增加字段

alter table employee modify gender char(2); # 修改字段的数据类型

alter table employee change nickname username varchar(30); # 修改字段名和字段的数据类型

alter table employee drop age; # 删除字段

alter table employee rename employee1; # 修改表名

truncate table employee1;

drop table if exists employee1;
