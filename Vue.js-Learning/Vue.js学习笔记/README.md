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

# 设置npm默认的安装位置
npm config set prefix "D:\programfiles\nodejs\node_global"

# 设置npm默认的缓存位置
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

### 项目初始化和安装

```cmd
# 初始化一个新的 npm 项目
npm init

# 安装 package.json 中的所有依赖
npm install

# 安装指定包
npm install 包名

# 安装并添加到 dependencies
npm install 包名 --save

# 安装并添加到 devDependencies
npm install 包名 --save-dev
```

在 package.json 文件中，主要有以下两种依赖：

- **dependencies**：
   - 项目运行时需要的依赖。

- **devDependencies**：
   - 在开发时需要的依赖，生产环境不需要。
   - 使用 `npm install <包名> --save-dev` 安装
   - 包括测试工具、构建工具等。


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
