import torch

# 1.基本自动微分示例
def basic_auto_grad_example():
    x = torch.tensor([2.0],requires_grad=True)
    y = x**2 + 2*x +1

    # 计算梯度
    y.backward() #反向传播
    print("y关于x的梯度：",x.grad) #梯度为6
basic_auto_grad_example()

# 2.使用 detach 分离计算图
def detach_example():
    x = torch.tensor([2.0],requires_grad=True)
    y = x**2
    z = y.detach() # 通常用于创建不需要梯度的副本
    w = z**3
    print("w是否需要梯度：",w.requires_grad) # false
detach_example()

# 3.梯度积累与清零
def grad_accumulation_example():
    x = torch.tensor([1.0],requires_grad=True)
    y1 = x**2
    y2 = x**3
    y1.backward()
    print(f"第一次的梯度：{x.grad}") # 2
    y2.backward()
    print(f"积累后的梯度：{x.grad}") # 2+3=5
    x.grad.zero_()
    print(f"清零后的梯度：{x.grad}") # 0
grad_accumulation_example()

# 4.禁用梯度追踪
def no_grad_example():
    x = torch.tensor([2.0],requires_grad=True)

    # 使用 torch.no_grad() 上下文管理器
    with torch.no_grad():
        y = x**2
        print(f"查看y是否需要梯度：{y.requires_grad}") # false
no_grad_example()

# 5.梯度截断
def grad_clip_example():
    x = torch.tensor([10.0],requires_grad=True)
    y = x**2
    y.backward()

    # 梯度截断
    torch.nn.utils.clip_grad_norm_(x, 5)
    print(f"截断后的梯度：{x.grad}")
grad_clip_example() #防止梯度爆炸
