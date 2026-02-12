"""
TCP客户端开发流程：
1、创建客户端socket对象
2、连接服务器端，指定：服务端IP，端口号
3、接收服务器端信息
4、给服务端发送消息
5、释放资源
"""

import socket

# 创建客户端socket对象
client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

# 连接服务器端，指定：服务端IP，端口号
client_socket.connect(("127.0.0.1",10086)) #参数格式：(地址，端口号)

# 接收服务器端信息
data = client_socket.recv(1024).decode("utf-8")
print(f"客户端接收到消息：{data}")

# 给服务端发送消息
client_socket.send("Hello Socket".encode("utf-8")) # 字符串编码为utf-8

# 释放资源
client_socket.close()
