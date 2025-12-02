# 什么是 vue3

vue3 是一个开发框架，就像 pytorch 之于 python 。

使用时需要安装，建议局部安装以和系统环境隔离。

# npm 包管理器

node.js的包管理器，就像 Pyhton 的 pip 一样。

## 常用指令(vue3 开发)

### 安装位置和缓存

```cmd
# 获取 npm 全局安装地址
npm config get prefix

# 获取 npm 缓存位置
npm config get cache

# 设置npm默认的安装位置
npm config set prefix "D:\nodejs\node_global"

# 设置npm默认的缓存位置
npm config set cache "D:\nodejs\node_cache"
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

### 安装依赖

在使用 npm 安装依赖时，默认是将依赖安装到当前项目的 node_modules 文件夹中。
除非你指定全局安装（使用 `-g` 标志）。

```cmd
# 安装 package.json 中的所有依赖
npm install

# 安装指定包
npm install 包名

npm install -g 包名 # 全局安装

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

### 依赖管理

```cmd

npm update # 更新所有依赖包

npm update 包名 # 更新指定的包

npm outdated # 检查过时的包

npm uninstall 包名 # 卸载指定的包

npm list # 列出项目中所有已安装的包

npm update -g # 更新所有全局安装的依赖包

npm update -g 包名 # 更新指定的全局安装的包

npm outdated -g # 检查过时的全局安装的包

npm uninstall -g 包名 # 卸载指定的全局安装的包

npm list -g # 列出所有全局安装的的包
```

### 创建 vue3 项目

```cmd
# 通过create-vue 搭建 vue3 项目
npm init vue@latest 项目名# 在当前工作目录下创建，注意切换工作目录

# 通过 vite 搭建 vue3 项目，同样在当前工作目录下创建
npm create vite@latest 项目名
```

说明：`@` 后面可指定vue版本号，latest 表示最新版

### vue3 项目开发

```cmd
npm run dev` # 启动开发服务器
`npm run build` # 构建生产版本
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

# 什么是 TypeScript

TypeScript 本质上就是 JavaScript 的“带静态类型系统的超集”。(TypeScript is a typed superset of JavaScript that compiles to plain JavaScript.)

TypeScript = JavaScript + 类型系统 + 一些语言增强

类型检查只在开发阶段和编译阶段发生，运行时不会存在类型信息。编译之后，所有的类型信息都会被全部擦除，不会留在产出的 .js 文件中。

除了类型，TypeScript 还增强了语言能力，例如：

1、接口(interface)、类型别名(type)

2、枚举(enum)

3、泛型(generics)

4、元组(tuple)

5、访问修饰符(public/private/protected)

6、装饰器(decorator)支持

7、类型推断、条件类型、映射类型等高级类型系统功能
