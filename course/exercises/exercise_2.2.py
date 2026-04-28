#Data
import numpy as np

# Land Area
x = np.array([122, 173, 341, 439, 477, 597, 682, 794, 912, 1001, 1081, 1203, 1255, 1355, 1465, 1650, 1707, 1768, 1863, 2015, 2133, 2158, 2282, 2423, 2496, 2575, 2679, 2846, 2890, 3026], dtype=float)

# Distance to city
d = np.array([12.9, 18.4, 11, 19.3, 19.7, 7.1, 10.2, 10, 0.1, 5.7, 5.8, 2.5, 3, 9.8, 2.5, 12.9, 12.2, 18.4, 3, 5.9, 6.2, 14.6, 14.9, 16.7, 17.9, 3.8, 19.6, 9.8, 2, 4.4], dtype=float)

# Price 
y = np.array([15.7, 11.3, 42, 35, 37.7, 75.5, 77.1, 88.7, 122, 119.1, 125.6, 145.5, 150, 145, 172, 170, 177, 169.4, 211, 220, 231.4, 216.9, 227.5, 239.4, 244.3, 279.8, 259.5, 295.4, 314.1, 324.7], dtype=float)

#Training the Linear Regression Model using Gradient Descent
#h(x) = z = a*x+b

#Initialize the parameters
a = 0
b = 0

alpha1 = 0.0000002
alpha2 = 0.004   

# Using Land Area as the feature
print(" Using Land area as the feature")

for i in range(100):
  z1 = a*x+b
  J1 = ((z1-y)**2).mean()

  print('i: %d, Cost: %f, a: %f, b: %f' % (i, J1, a, b))

  grad_a = ((z1-y)*x).mean()*2 # (Patial derivative of the cost function with respect to a)
  grad_b = (z1-y).mean()*2     # (Patial derivative of the cost function with respect to b)

  a += -alpha1*grad_a
  b += -alpha1*grad_b

  #Evaluating the model

print("Verify the model")

z1 = a*x+b

for zi, yi in zip(z1, y):
  print('Predicted: %.2f, Actual: %.2f' % (zi, yi))

# Using Distance to city as the feature 
print(" Using distance to city as the feature")

# reset a,b

a = 0
b = 0

for i in range(100):
  z2 = a*d+b
  J2 = ((z2-y)**2).mean()

  print('i: %d, Cost: %f, a: %f, b: %f' % (i, J2, a, b))

  grad_a = ((z2-y)*d).mean()*2 # (Patial derivative of the cost function with respect to a)
  grad_b = (z2-y).mean()*2     # (Patial derivative of the cost function with respect to b)

  a += -alpha2*grad_a
  b += -alpha2*grad_b

  #Evaluating the model

print("Verify the model\n")

z2 = a*d+b

for zi, yi in zip(z2, y):
  print('Predicted: %.2f, Actual: %.2f' % (zi, yi) )

# ==================== Exercise 2 ========================

print("Using Polynomial hypothesis \n")

x2 = np.array([
    129, 214, 343, 387, 454, 565, 675, 798, 879, 1016,
    1095, 1222, 1272, 1429, 1467, 1593, 1661, 1773, 1946,
    2046, 2115, 2235, 2337, 2391, 2516, 2564, 2669, 2781,
    2875, 3001], dtype=float)
x_sqrt = np.sqrt(x2)

theta0 = 80.0
theta1 = -20.0
theta2 = 90.0

alpha3 = 0.006

for i in range(100):
    
    # Hypothesis / prediction
    h = theta0 + theta1 * x + theta2 * x_sqrt
    
    # Error = prediction - actual
    error = h - y
    
    # Cost (mean squared error)
    J3 = np.mean(error ** 2)

    print('i: %d, Cost: %f, theta1: %f, theta2: %f, theta3: %f' % (i, J3, theta0, theta1, theta2))
    
    # Gradients (partial derivatives) - we include the factor 2
    grad_theta0 = np.mean(error) * 2
    grad_theta1 = np.mean(error * x) * 2
    grad_theta2 = np.mean(error * x_sqrt) * 2

    # Update parameters (gradient descent step)
    theta0 = theta0 - alpha3 * grad_theta0
    theta1 = theta1 - alpha3 * grad_theta1
    theta2 = theta2 - alpha3 * grad_theta2

print("Verify the model\n")

h = theta0 + theta1 * x + theta2 * x_sqrt

for hi, yi in zip(h, y):
  print('Predicted: %.2f, Actual: %.2f' % (hi, yi))
