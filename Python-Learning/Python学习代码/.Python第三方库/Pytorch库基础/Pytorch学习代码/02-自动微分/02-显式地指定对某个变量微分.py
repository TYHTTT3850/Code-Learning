"""
使用 torch.autograd.grad()求表达式对指定变量的梯度
torch.autograd.grad(
    outputs,
    inputs,
    grad_outputs=None,
    retain_graph=False,
    create_graph=False,
    only_inputs=True,
    allow_unused=False
)
outputs	        要从它开始反向传播的结果表达式(就是要求导的表达式)
inputs	        你希望求梯度的变量，必须是 requires_grad=True 的张量
grad_outputs	指定外部梯度(即复合函数求导的外部导数值)
retain_graph	是否保留计算图(默认释放)
create_graph	是否构建计算梯度的计算图(用于高阶导数)
only_inputs	    是否只对 inputs 求导(默认 True)
allow_unused	如果某个 input 没有参与计算，是否允许其梯度为 None
"""
import torch

# 基本的使用
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
grad = torch.autograd.grad(y, x)[0]
print(grad)  # 输出：tensor([4.])

# 高阶导数
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3      # dy/dx = 3x^2
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(d2y_dx2)  # 输出：12.0 (即 6x，x=2 时为12)

# 多个输入求导
x = torch.tensor(1.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

y = w * x + b
grads = torch.autograd.grad(y, [x, w, b])
print(grads)  # 输出：(tensor(2.), tensor(1.), tensor(1.))

# 含外部导数的求导
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = x ** 2

# 假设我们从某个函数 z = f(y) 得到了 ∂z/∂y
# 用一个和 y 形状相同的“上游梯度”来模拟
grad_output = torch.ones_like(y) * 2.0 #外部导数

# 计算 ∂z/∂x = ∂z/∂y × ∂y/∂x
#y = x ** 2，所以 ∂y/∂x = 2x
#grad_outputs = 2，所以链式法则变成：2 * 2x = 4x
grad = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_output)[0]
print(grad) # [[4,8],[12,16]]


