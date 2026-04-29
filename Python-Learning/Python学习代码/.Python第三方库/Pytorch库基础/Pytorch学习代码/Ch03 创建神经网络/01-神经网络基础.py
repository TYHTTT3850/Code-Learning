"""
神经网络类型
1、前馈神经网络（Feedforward Neural Networks）：数据单向流动，从输入层到输出层，无反馈连接。
2、卷积神经网络（Convolutional Neural Networks, CNNs）：适用于图像处理，使用卷积层提取空间特征。
3、循环神经网络（Recurrent Neural Networks, RNNs）：适用于序列数据，如时间序列分析和自然语言处理，允许信息反馈循环。
4、长短期记忆网络（Long Short-Term Memory, LSTM）：一种特殊的RNN，能够学习长期依赖关系。
"""

import torch

# 简单的全连接神经网络
class SimpleNN(torch.nn.Module):
    """
    torch.nn.Module 是所有神经网络模块的基类，你需要定义以下两个部分：
    __init__()：构造函数，定义网络层。
    forward()：定义数据的前向传播过程。为训练时使用模型输出预测值提供支持。
    PyTorch 提供了许多常见的神经网络层，以下是几个常见的：
        1、torch.nn.Linear(in_features, out_features)：全连接层，输入 in_features 个特征，输出 out_features 个特征。
        2、nn.Conv2d(in_channels, out_channels, kernel_size)：2D 卷积层，用于图像处理。
        3、torch.nn.MaxPool2d(kernel_size)：2D 最大池化层，用于降维。
    """
    def __init__(self):
        super().__init__() # 调用父类构造函数
        # 定义一个输入层到隐藏层的全连接层
        self.fc1 = torch.nn.Linear(2, 2)  # 输入 2 个特征，输出 2 个特征
        # 定义一个隐藏层到输出层的全连接层
        self.fc2 = torch.nn.Linear(2, 1)  # 输入 2 个特征，输出 1 个预测值

    def forward(self,x):
        x = torch.nn.functional.relu(self.fc1(x)) #使用ReLU激活函数
        x = self.fc2(x)
        return x

model = SimpleNN.nn() # 创建模型实例
print(model)

"""
激活函数（Activation Function）
激活函数决定了神经元是否应该被激活。它们是非线性函数，使得神经网络能够学习和执行更复杂的任务。常见的激活函数包括：
1、Sigmoid：用于二分类问题，输出值在 0 和 1 之间。
2、Tanh：输出值在 -1 和 1 之间，常用于输出层之前。
3、ReLU（Rectified Linear Unit）：目前最流行的激活函数之一，定义为 f(x) = max(0, x)，有助于解决梯度消失问题。
4、Softmax：常用于多分类问题的输出层，将输出转换为概率分布。
"""

input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# 模块式
module_output1 = torch.nn.ReLU()
module_output2 = torch.nn.Tanh()
module_output3 = torch.nn.Sigmoid()

# 函数式
output1 = torch.nn.functional.relu(input_tensor)# ReLU 激活
output2 = torch.nn.functional.sigmoid(input_tensor)# Sigmoid 激活
output3 = torch.nn.functional.tanh(input_tensor)# Tanh 激活

"""
损失函数（Loss Function）
损失函数用于衡量模型的预测值与真实值之间的差异。
常见的损失函数包括：
1、均方误差（MSELoss）：回归问题常用，计算输出与目标值的平方差。
2、交叉熵损失（CrossEntropyLoss）：分类问题常用，计算输出和真实标签之间的交叉熵。
3、BCEWithLogitsLoss：二分类问题，结合了 Sigmoid 激活和二元交叉熵损失。
"""

# 模块式
module_criterion1 = torch.nn.MSELoss()
module_criterion2 = torch.nn.CrossEntropyLoss()
module_criterion3 = torch.nn.BCEWithLogitsLoss()

# 函数式
y_pred = torch.tensor([1.0])
y_true = torch.tensor([0.0])


loss1 = torch.nn.functional.mse_loss(y_pred, y_true)
loss2 = torch.nn.functional.cross_entropy(y_pred, y_true)
loss3 = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)

"""
优化器（Optimizer）
优化器负责在训练过程中更新网络的权重和偏置。
常见的优化器包括：
SGD（随机梯度下降）
Adam（自适应矩估计）
RMSprop（均方根传播）
"""
# 使用 SGD 优化器
optimizer1 = torch.optim.SGD(model.parameters(), lr=0.01)

# 使用 Adam 优化器
optimizer2 = torch.optim.Adam(model.parameters(), lr=0.001)
