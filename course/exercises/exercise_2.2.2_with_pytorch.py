import pandas as pd
import numpy as np
import torch
import torch.optim as optm

data = pd.read_csv("/home/tikea/ecam_ws/ecam_courses/machine_learning/land_price_2.csv").to_numpy()
print(data.shape)

x = data[:, 0]
y = data[:, 1]

x_sqrt        = np.sqrt(x)
x_scaled      = (x-x.mean())/x.std()
x_sqrt_scaled = (x_sqrt-x_sqrt.mean())/x_sqrt.std()

tx       = torch.tensor(x_scaled,      dtype=torch.float32)
tx_sqrot = torch.tensor(x_sqrt_scaled, dtype=torch.float32)
ty       = torch.tensor(y,             dtype=torch.float32)

# Build the hypothesis

theta0 = torch.tensor(0, dtype=torch.float32, requires_grad=True)
theta1 = torch.tensor(0, dtype=torch.float32, requires_grad=True)
theta2 = torch.tensor(0, dtype=torch.float32, requires_grad=True)

opt = optm.SGD([theta0, theta1, theta2], lr=0.5)

for i in range(1000):
    tz = theta0 + theta1*tx + theta2*tx_sqrot
    J  = ((tz - ty)**2).mean()

    print("i: %d, cost: %f" % (i, J.item()))

    J.backward()

    opt.step()
    opt.zero_grad()

# Evaluate the model

tz = theta0 + theta1*tx + theta2*tx_sqrot

for zi, yi in zip(tz, ty):
  print('Predicted: %.2f, Actual: %.2f' % (zi, yi))

