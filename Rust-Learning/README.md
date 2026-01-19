检测 Rust 环境是否正确安装

```cmd
rustc -V
cargo -V
```

## Cargo

Cargo 是 Rust 的官方构建系统和包管理器。它主要有两个作用：

主要有两个作用：**项目管理**、**包管理器**。

Cargo 项目管理命令：

```cmd
cargo new <project-name> #创建一个新的 Rust 项目。
cargo build #编译当前项目。
cargo run #编译并运行当前项目。
cargo check #检查当前项目的语法和类型错误。
cargo test #运行当前项目的单元测试。
cargo update #更新 Cargo.toml 中指定的依赖项到最新版本。
cargo --help #查看 Cargo 的帮助信息。
cargo publish #将 Rust 项目发布到 crates.io。
cargo clean #清理构建过程中生成的临时文件和目录。
```

使用Cargo生成的Rust 项目的最小结构：

```
RustProject/
├── Cargo.toml    # 定义项目元信息、依赖和构建配置
├── Cargo.lock    # 锁定依赖的具体版本以保证构建可复现
└── src/          # 源代码
    └── main.rs   # 程序入口文件，包含 main 函数
```