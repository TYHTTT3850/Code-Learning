# DQL 小结

DQL 英文全称是 Data Query Language (数据查询语言)，用来查询数据库中表的记录。

查询关键字：SELECT

在一个正常的业务系统中，查询操作的频次是要远高于增删改的，当我们去访问企业官网、电商网站，在这些网站中我们所看到的数据，实际都是需要从数据库中查询并展示的。而且在查询的过程中，可能还会涉及到条件、排序、分页等操作。

基本语法：

```sql
SELECT
    字段列表
FROM
    表名列表
WHERE
    条件列表
GROUP BY
    分组字段列表
HAVING
    分组后条件列表
ORDER BY
    排序字段列表
LIMIT
    分页参数
```

基本查询(不带任何条件)

条件查询(WHERE)

聚合函数(count、max、min、avg、sum)

分组查询(group by)

排序查询(order by)

分页查询(limit)

### 执行顺序

DQL语句的执行顺序为：from ... where ... group by ... having ... select ... order by ... limit ...
