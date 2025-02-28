# `systemctl` 命令

Linux系统很多软件(内置或第三方)均支持使用 `systemctl` 命令控制：启动，停止，开机自启。

能够被 `systemctl` 管理的软件也称之为：服务。语法：

```bash
systemctl {start | stop | status | enable | disable} {服务名}
```

`start` ：启动。

`stop` ：关闭。

`status` ：查看状态。

`enable` ：开启开机自启。

`disable` ：关闭开机自启。

系统内置的服务比较多，比如：

NetworkManager：主网络服务。

network：副网络服务。

firewalld：防火墙。

sshd：ssh服务(FinalShell远程登录用的就是这个服务)。

**注意**：

1、系统内置服务均可被 `systemctl` 控制。

2、第三方软件如果自动注册为系统服务，则可被 `systemctl` 控制。

3、第三方软件如果没有注册为系统服务。则可以手动注册。