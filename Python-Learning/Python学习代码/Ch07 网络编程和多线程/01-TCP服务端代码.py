"""
网络编程也叫 socket 通信，即通信双方都有一个 socket 对象。
数据在 socket 之间通过数据报包(UDP)或者字节流(TCP)的方式进行传输。
"""

import socket
"""
参数解释：
1、family：地址簇，AF_INET表示IPv4，AF_INET6表示IPv6，AF_UNIX表示本地通信
2、type：套接字类型，SOCK_STREAM表示TCP，SOCK_DGRAM表示UDP，SOCK_RAW表示原始套接字
"""
socket_obj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)# 创建socket对象

"""
TCP服务端开发流程：
1、创建服务端socket对象
2、绑定IP地址和端口号
3、设置最大监听数
4、等待客户端申请建立连接
5、给客户端发送消息
6、接收客户端发送的消息
7、关闭连接
"""

# 创建服务端socket对象
server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

# 绑定IP地址和端口号
server_socket.bind(("127.0.0.1",10086)) #参数格式：(地址，端口号)

# 设置最大监听数
server_socket.listen(5) #参数表示最大监听数

# 等待客户端申请建立连接
print(1)
accept_socket,client_info = server_socket.accept() #返回值是一个元组,格式为(负责和客户端交互的socket,客户端的信息)
print(2) # 2不会输出，因为卡在accept()了

# 给客户端发送消息
accept_socket.send(b"Welcome To Socket") #字符串前加b表示转换成二进制

# 接收客户端发送的消息
data = accept_socket.recv(1024).decode("utf-8") #1024表示一次接收的最大字节数,decode("utf-8")表示将二进制转换成字符串
print(data)

# 释放资源
accept_socket.close() #关闭负责和客户端交互的socket
# server_socket.close() #服务器端一般不关闭
