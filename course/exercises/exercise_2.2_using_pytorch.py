import pandas as pd
import numpy as np
import torch

data = pd.read_csv("/home/tikea/ecam_ws/ecam_courses/machine_learning/land_price_1.csv").to_numpy()
print(data.shape)

x1 = data[:, 0]
x2 = data[:, 1]
y  = data[:, 2]

x1_scaled = (x1-x1.mean())/x1.std()
x2_scaled = (x2-x2.mean())/x2.std()

tx1 = torch.tensor(x1_scaled, dtype=torch.float32)
tx2 = torch.tensor(x2_scaled, dtype=torch.float32)
ty  = torch.tensor(y        , dtype=torch.float32)

alpha = 0.1

# Build the hypothesis

a = torch.tensor(0, dtype=torch.float32, requires_grad=True)
b = torch.tensor(0, dtype=torch.float32, requires_grad=True)
c = torch.tensor(0, dtype=torch.float32, requires_grad=True)

for i in range(50):
    tz = a*tx1 + b*tx2 + c
    J  = ((tz - ty)**2).mean()

    print("i: %d, cost: %f" % (i, J.item()))

    J.backward()

    with torch.no_grad():
        a += -alpha*a.grad
        b += -alpha*b.grad
        c += -alpha*c.grad

        # reset the gradiant
        a.grad.zero_()
        b.grad.zero_()
        c.grad.zero_()

# Evaluate the model

tz = a*tx1 + b*tx2 + c 


for zi, yi in zip(tz, ty):
  print('Predicted: %.2f, Actual: %.2f' % (zi, yi))