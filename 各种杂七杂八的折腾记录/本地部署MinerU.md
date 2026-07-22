# MinerU主要功能

PDF(原生、扫描件、乱码件)、Office 文档(DOCX、PPTX、XLSX)以及常见图像(PNG、JPG 等)的全格式输入。

在解析后，自动剔除页眉页脚等噪点，恢复正确的阅读顺序，并将文档统一输出为包含 LaTeX 公式和 HTML 表格的 Markdown 文件、按阅读逻辑切分的结构化 JSON 数据，并能够独立抽取图片素材。

# 本地部署

建议在项目根目录下新建一个 Python 虚拟环境，从而和系统中的其他环境隔离，注意不要有中文路径。

在项目目录中打开终端，并确认进入虚拟环境，输入：

```cmd
pip install -U "mineru[all]"
```

默认会安装 CPU 版 PyTorch，若需要启用 GPU 加速，则需要安装 GPU 版 Pytorch：

```cmd
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

安装完成后，输入：

```cmd
mineru --version
```

查看版本，并检验是否安装成功，若成功，则会跳出类似这样的信息：`mineru, version 3.4.4`

然后进行第一次运行：

```cmd
mineru -p "你的文件.pdf" -o "输出路径"
```

第一次运行时需要从远程下载模型，默认的下载路径是：

```
C:\Users\<你的用户名>\.cache\huggingface
或者
C:\Users\<你的用户名>\.cache\modelscope
具体在哪取决于下载源
```

如果第一次运行失败，可能是无法使用 `lmdeploy` 底层加速库，卸载即可(不报错就忽略)：

```cmd
pip uninstall lmdeploy -y
```

可以剪切下载的模型到指定位置，并在 `C:\Users\<你的用户名>` 下找到 `mineru.json` 修改模型路径到指定位置，例如：

```json
"models-dir": {
        "pipeline": "D:\\MinerU_Models\\models--opendatalab--PDF-Extract-Kit-1.0\\snapshots\\ed6b654c018d742e65a17671e379c5e6ecc87ec9",
        "vlm": "D:\\MinerU_Models\\models--opendatalab--MinerU2.5-Pro-2605-1.2B\\snapshots\\bff20d4ae2bf202df9f45284b4d43681555a97ed"
    },
    "model-source": "huggingface",
    "config_version": "1.3.2"
}
```

未来 MinerU 发布了更强的新版本和新模型时的更新步骤：

1. 升级 MinerU 核心代码：

```cmd
pip install -U "mineru[all]"
```

可能会把 `lmdeploy` 底层加速库安装回来导致报错，所以同样需要删除(不报错就忽略)。

2. 下载新模型

在 `mineru.json` 中临时把模型路径改为一个不存在的目录或空目录，例如：

```json
"models-dir": "D:\\Temp_Download"
```

然后运行一次解析命令从而触发重新下载，并将模型放到默认的下载路径中：

```cmd
mineru -p "你的文件.pdf" -o "输出路径"
```

最后将新的模型移入指定位置，并在 `mineru.json` 中配置新的模型路径。
