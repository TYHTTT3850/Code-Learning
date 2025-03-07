修改数据的具体语法为：

```sql
UPDATE 表名 SET 字段名1 = 值1 , 字段名2 = 值2 , .... [ WHERE 条件 ] ;
```

例如：

A. 修改id为1的数据，将name修改为test。

```sql
update employee set name = 'test' where id = 1;
```

B. 修改id为1的数据, 将name修改为test1, gender修改为女

```sql
update employee set name = '小昭' , gender = '女' where id = 1;
```

C. 将所有的员工入职日期修改为 2008-01-01

```sql
update employee set entrydate = '2008-01-01';
```

**注意**：

修改语句的条件可以有，也可以没有，如果没有条件，则会修改**整张**表的**所有**数据。  

