# 项目文件介绍
public文件夹：存放着网页的标签图标

src文件夹：源代码文件

env.d.ts：用来给项目中用到的环境变量添加类型，让 TypeScript 能识别并校验这些变量。

index.html：网页的入口文件

package.json：项目所需的依赖

tsconfig.app.json tsconfig.json tsconfig.node.json：均是TypeScript的配置文件

vite.config.ts：项目的配置文件，基于 vite 的配置

## src 文件夹内容解析
main.ts：创建应用实例

App.vue：组件文件，网页应用的根文件

components文件夹：存放其他的(树叶)组件

assets文件夹：存放 css 样式，svg 图片等内容