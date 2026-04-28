# torch is very good because it calculate patial derivative for u
import torch
import numpy as np

a = 10
ta = torch.tensor(a, dtype=torch.float32)

b = [1, 2, 3]
tb = torch.tensor(b, dtype=torch.float32)

print("a", a)
print("ta", ta)

c = np.random.randn(2, 3)
tc = torch.tensor(c, dtype=torch.float32)

a = torch.tensor(10, dtype=torch.float32)
a.requires_grad = True
b = torch.tensor(20, dtype=torch.float32, requires_grad=True)
x = torch.tensor(100)
y = torch.tensor(200)

z = a*x + b
J = (z-y) ** 2

J.backward()

grad_a = a.grad
grad_b = b.grad

print("grad_a = ", grad_a ,"\n")
print("grad_b = ", grad_b)
