# R语言注意事项

赋值符不是 `=` 而是 `<-`。

安装指定的包：

```R
install.packages("包名")
```

安装完指定包后会有压缩包残留，可以删除，若不想有残留，可以在安装时指定：

```R
install.packages("包名",clean=TRUE)
```

删除指定包：

```R
remove.packages("包名")
```

查看包的安装路径，输入命令

```R
.libPaths()
```
