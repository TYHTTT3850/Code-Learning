# 什么是命令，命令行

命令行：即Linux终端(Terminal)，可以使用各种字符化命令对系统发出操作指令。

命令：即Linux程序，一个命令就是一个Linux的程序，也就是说：命令的本体是二进制可执行程序。命令没有图形化界面，可以在命令行(终端)中提供字符化反馈。

# Linux命令基础格式

无论什么命令，都有类似于格式：`command [-options] [parameter]` 。

一般约定，在命令的格式说明中 `[]` 表示可选，`{}` 表示必选，`|` 表示或。在实际输入命令时，`[]` 、`{}` 、`|` 不用真的出现，只是一种表示形式而已。

`command` ：命令本身。

`[-options]` ：命令的一些选项，可以通过选项控制命令的行为细节。

`[paremeters]` ：命令的参数，多数用于命令的指向目标。

如果命令有多个选项可以一起使用的，则可以写在一起，不需要分开写。例如：

```bash
ls -a -l -h
```

选项写在一起也可以：

```bash
ls -alh
```

选项对大小写敏感

若无特殊要求，选项的顺序可以随意调换。

如果命令可以使用多个参数，一般用空格隔开。


