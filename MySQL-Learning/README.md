### 启动 MySQL 服务

MySQL 安装完成后，在系统启动时，会自动启动 MySQL 服务，无需手动启动。

若需要手动控制，以**管理员身份**执行以下指令：

```cmd
net start mysql80

net stop mysql80
```

mysql80 指的是再安装 MySQL 时，默认指定的系统服务名称，不是固定的，可以自己修改，若没有修改过，则默认是 mysql80 。

### 客户端连接

方式一：通过 MySQL 提供的客户端命令行工具 MySQL Command Line Client

方式二：使用系统自带的命令行工具，执行下述命令：

```cmd
mysql [-h MySQL server ip] [-P port] -u root -p
```

参数解释：

`-h`（Host）：

- 指定 MySQL 服务器的 IP 地址。

- 如果省略，默认为本地主机(localhost)。

- 示例：`-h 192.168.1.100` 。

`-P`（Port）：

- 指定连接的端口号。

- MySQL 默认端口是 3306。

- 大写 P。

- 示例：`-P 3306` 。

`-u`（User）：

- 指定登录用户名。

- 这里是 `root`，代表管理员用户。

- 大小写敏感。

`-p`（Password）：

- 表示需要输入密码。

- 小写 p。

- 执行后会提示交互式输入密码。

- 输入密码时不会显示字符，增加安全性。