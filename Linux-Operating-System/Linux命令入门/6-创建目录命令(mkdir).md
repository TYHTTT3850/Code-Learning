`mkdir` 命令来自英文：Make Directory

语法如下：

```bash
mkdir [-p] {Linux路径}
```

`-p` ：表示自动创建不存在的父目录，适用于创建连续多层级目录。例如：

```bash
mkdir test/subtest/subsubtest
```

这条指令如果上级目录不存在，则无法创建，会报错。

加上 `-p` 选项后，即可将一整个链条都创建完成。

```bash
mkdir -p test/subtest/subsubtest
```

**注意**：创建文件夹需要修改权限，如无特殊需求，请确保操作均在用户HOME目录内，若在此之外操作会涉及到权限问题。
