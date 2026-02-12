"""
数据预处理和增强对于提高模型的性能至关重要。
PyTorch 提供了 torchvision.transforms 模块来进行常见的图像预处理和增强操作，如旋转、裁剪、归一化等。
"""

# 常见的图像预处理操作
import torchvision.transforms as transforms #图像预处理
from PIL import Image #图像加载

# 定义数据预处理的流水线
transform1 = transforms.Compose([
    transforms.Resize((128,128)), # 图像调整为 128×128
    transforms.ToTensor(), # 图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) # 标准化
    ])

# 示例图片，实际开发中图片是从文件夹中加载而来
image = Image.open("image.jpeg")

# 应用预处理
image_tensor = transform1(image)
print(image_tensor.shape) #输出张量形状 torch.Size([3, 128, 128])

# 图像数据增强技术通过对训练数据进行随机变换，增加数据的多样性，帮助模型更好地泛化。例如，随机翻转、旋转、裁剪等。
# 常见的数据增强方法
transform2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),# 随机水平翻转
    transforms.RandomRotation(30),# 随机翻转30度
    transforms.RandomResizedCrop(128),# 随机裁剪并调整大小为 128×128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
