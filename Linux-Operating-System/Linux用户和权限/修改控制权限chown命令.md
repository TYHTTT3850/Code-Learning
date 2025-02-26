# `chown` 命令

修改文件或文件夹的所属用户和用户组。语法：

```bash
chown [-R] [用户] [:] [用户组] 文件或文件夹
```

`-R` ：同 `chmod` 命令，对文件夹内所有内容应用相同的规则。

`[:]` ：表示分隔符。

**示例**：

```bash
chown root hello.txt #仅修改所属用户

chown :root hello.txt #仅修改所属用户组

chown root:user hello.txt #同时修改

chkwn -R root test #对文件夹内所有内容应用相同规则
```

**注意**：普通用户无法修改文件、文件夹所属为其他用户或组，所以此命令之有root用户能执行。