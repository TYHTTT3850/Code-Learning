create table emp(
num int comment '编号',
id_work varchar(10) comment '员工工号',
name varchar(10) comment '员工姓名',
gender char(1) comment '性别',
age tinyint unsigned comment '年龄',
id_number char(18) comment '身份证号',
join_time date comment '入职时间'
) comment '员工表';