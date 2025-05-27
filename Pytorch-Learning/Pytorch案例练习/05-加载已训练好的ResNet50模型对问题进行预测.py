import torch
import os
from PIL import Image
from torchvision.models import resnet50
from torchvision import transforms
import matplotlib.pyplot as plt

# 基本配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './best_model.pth'
image_folder = './problem1/'
num_classes = 4

# 加载模型
model = resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# 定义预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 收集所有图片路径
img_paths = []
filenames = os.listdir(image_folder)

for filename in filenames:
    filename_lower = filename.lower()
    if filename_lower.endswith('.jpg') or filename_lower.endswith('.jpeg') or filename_lower.endswith('.png'):
        full_path = os.path.join(image_folder, filename)
        img_paths.append(full_path)

# 处理所有图片
images = []

for path in img_paths:
    img = Image.open(path)
    img = transform(img)
    images.append(img)

batch = torch.stack(images)
batch = batch.to(device)

# 推理预测
with torch.no_grad():
    outputs = model(batch)
    _, preds = torch.max(outputs, dim=1)

# 打印结果
classes = ["COVID","LungOpacity","Normal","ViralPneumonia"]
for i in range(len(img_paths)):
    filename = os.path.basename(img_paths[i])
    pred_class = preds[i].item()
    print(f"{filename} -> 类别: {classes[pred_class]}")

# 图像总数
n_images = len(images)
cols = 4  # 每行显示的图片数
rows = (n_images + cols - 1) // cols  # 自动计算行数

plt.figure(figsize=(cols * 4, rows * 4))

for i in range(n_images):
    img = images[i].cpu().permute(1, 2, 0)  # 将张量转换为 HWC 格式
    img = img * 0.5 + 0.5  # 反归一化

    plt.subplot(rows, cols, i + 1)
    plt.imshow(img.numpy())
    plt.title(f"{classes[preds[i].item()]}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig("肺炎识别预测结果")
plt.show()
