use test;

create table emp(id int comment '编号',
                 workno varchar(10) comment '工号',
                 name varchar(10) comment '姓名',
                 gender char(1) comment '性别',
                 age tinyint unsigned comment '年龄',
                 idcard char(18) comment '身份证号',
                 workaddress varchar(50) comment '工作地址',
                 entrydate date comment '入职时间') comment '员工表';

insert into emp (id, workno, name, gender, age, idcard, workaddress, entrydate) values (1, '00001', '柳岩666', '女', 20, '123456789012345678', '北京', '2000-01-01');

insert into emp (id, workno, name, gender, age, idcard, workaddress, entrydate) values (2, '00002', '张无忌', '男', 18, '123456789012345670', '北京', '2005-09-01');

insert into emp (id, workno, name, gender, age, idcard, workaddress, entrydate) values (3, '00003', '韦一笑', '男', 38, '123456789712345670', '上海', '2005-08-01');

insert into emp (id, workno, name, gender, age, idcard, workaddress, entrydate) values (4, '00004', '赵敏', '女', 18, '123456757123845670', '北京', '2009-12-01');

INSERT INTO emp (id, workno, name, gender, age, idcard, workaddress, entrydate) VALUES (5, '00005', '小昭', '女', 16, '123456769012345678', '上海', '2007-07-01');

INSERT INTO emp (id, workno, name, gender, age, idcard, workaddress, entrydate) VALUES (6, '00006', '杨逍', '男', 28, '12345678931234567X', '北京', '2006-01-01');

INSERT INTO emp (id, workno, name, gender, age, idcard, workaddress, entrydate) VALUES (7, '00007', '范瑶', '男', 40, '123456789212345670', '北京', '2005-05-01');

INSERT INTO emp (id, workno, name, gender, age, idcard, workaddress, entrydate) VALUES (8, '00008', '黛绮丝', '女', 38, '123456157123645670', '天津', '2015-05-01');

INSERT INTO emp (id, workno, name, gender, age, idcard, workaddress, entrydate) VALUES (9, '00009', '范凉凉', '女', 45, '123156789012345678', '北京', '2010-04-01');

INSERT INTO emp (id, workno, name, gender, age, idcard, workaddress, entrydate) VALUES (10, '00010', '陈友谅', '男', 53, '123456789012345670', '上海', '2011-01-01');

INSERT INTO emp (id, workno, name, gender, age, idcard, workaddress, entrydate) VALUES (11, '00011', '张士诚', '男', 55, '123567897123465670', '江苏', '2015-05-01');

INSERT INTO emp (id, workno, name, gender, age, idcard, workaddress, entrydate) VALUES (12, '00012', '常遇春', '男', 32, '123446757152345670', '北京', '2004-02-01');

INSERT INTO emp (id, workno, name, gender, age, idcard, workaddress, entrydate) VALUES (13, '00013', '张三丰', '男', 88, '123656789012345678', '江苏', '2020-11-01');

INSERT INTO emp (id, workno, name, gender, age, idcard, workaddress, entrydate) VALUES (14, '00014', '灭绝', '女', 65, '123456719012345670', '西安', '2019-05-01');

INSERT INTO emp (id, workno, name, gender, age, idcard, workaddress, entrydate) VALUES (15, '00015', '胡青牛', '男', 70, '12345674971234567X', '西安', '2018-04-01');

INSERT INTO emp (id, workno, name, gender, age, idcard, workaddress, entrydate) VALUES (16, '00016', '周芷若', '女', 18, null, '北京', '2012-06-01');

update emp set name = 'test1' where id = 1;

update emp set name = 'test2' where id = 2;

update emp set name = 'test3' where id = 3;

update emp set name = 'test4' where id = 4;

update emp set name = 'test5' where id = 5;

update emp set name = 'test6' where id = 6;

update emp set name = 'test7' where name = '范瑶';

update emp set name = 'test8' where name = '黛绮丝';

update emp set name = 'test9' where name = '范凉凉';

update emp set name = 'test10',age=35 where name = '陈友谅';

update emp set name = 'test11', age=23, workaddress='浙江' where id = 11;

update emp set name = 'test12', age=27, gender = '女', workaddress='福建' where id = 12;

update emp set name = 'test13', age=80, gender = '男', workaddress='广东' where id = 13;

update emp set name = 'test14', age=56,workaddress='山东' where name = '灭绝';

update emp set name = 'test15', age=75 where workno = '00015';

update emp set name = 'test16', age=34 where workno = '00016';

delete from emp where gender = '男';

delete from emp where id = 1;

delete from emp where name = 'test4';

delete from emp;
