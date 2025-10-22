## 基本说明

jdk版本：jdk25(LST)。开发环境：Intellij IDEA

建议在环境变量中新建`JAVA_HOME`变量，里面的路径即为`jdk`安装路径。

然后在`path`环境变量中添加一个路径为`%JAVA_HOME\bin%`

Java项目构成：工程(Project)，模块(Module)，包(Package)，类(Class)。关系为：层层递进，层层包含

例子：**小区**(工程)里有**楼房**(模块)，**楼房**(模块)里有**楼层**(包)，**楼层**(包)里有**房间**(类)。

创建完模块后，在模块的`src`目录下创建包，一般包的命名规范：`域名倒写+技术名称`，如`com.aaa.技术名称`

运行程序后，会自动编译并把结果存入工程的`out`目录下(使用Intellij管理项目)。

Java声明变量时需要加上类型，和C++一样

## 使用Maven构建项目

在IDEA创建项目时，选择使用Maven构建系统。

创建项目后，最主要的是`src`目录、`pom.xml`文件。

​	1、`pom.xml`是项目的配置文件。

​	2、`src`目录存放程序源代码，其下有`main`目录和`test`目录。

​	3、`test`用于写测试程序，其下有`java`目录和`resources`目录，`java`存放代码，`resources`存放配		置文件

​	4、`main`用于写程序源码，其下有`java`目录和`resources`目录，`java`存放代码，`resources`存放配		置文件

编译后的结果在`target`目录下。