import torch

print(torch.cuda.is_available())

x = torch.rand(5, 3)
print(x)

print(torch.version.cuda)