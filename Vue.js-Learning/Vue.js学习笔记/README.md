[笔记配套视频](https://www.bilibili.com/video/BV1HV4y1a7n4/?p=3&share_source=copy_web&vd_source=5d1a88af6b151c4524e2e0393d9d7b02)，day 11开始为 vue3，往前为 vue2。这里只学习 vue3。

# npm 包管理器

就像 Pyhton 的 pip 一样。

## 常用指令

### 安装位置和缓存

```cmd
# 获取 npm 全局安装地址
npm config get prefix

# 获取 npm 缓存位置
npm config get cache

# 设置npm安装的地址
npm config set prefix "D:\programfiles\nodejs\node_global"

# npm缓存位置设置
npm config set cache "D:\programfiles\nodejs\node_cache"
```

### 下载源

npm 官方原始镜像网址是：https://registry.npmjs.org/

淘宝 NPM 镜像：http://registry.npmmirror.com

阿里云 NPM 镜像：https://npm.aliyun.com

腾讯云 NPM 镜像：https://mirrors.cloud.tencent.com/npm/

华为云 NPM 镜像：https://mirrors.huaweicloud.com/repository/npm/

```cmd
# 切换镜像网址
npm config set registry 镜像网址

# 查看当前镜像网址
npm config get registry
```

# nrm
nrm 是一个 npm 源管理器，允许你快速地在 npm 源间切换。

```cmd
# 安装nrm
npm install -g nrm

# 查看可选的源
nrm ls

# 切换。如果要切换到taobao源，执行命令
nrm use taobao

# 测试速度
nrm test
```
