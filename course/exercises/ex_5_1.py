import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv('/home/tikea/ecam_ws/ecam_courses/machine_learning/data_3_1_2.csv').to_numpy()
x = data[:, :-1]
y = data[:, -1]

#feature scaleing
x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

m_train = int(0.6 * x.shape[0])
x_train = x[:m_train]
y_train = y[:m_train]
x_test = x[m_train:]
y_test = y[m_train:]

model = LogisticRegression()
model.fit(x_train, y_train)

#Evaluation

TP = np.sum((model.predict(x_test) == 1) & (y_test == 1))
TN = np.sum((model.predict(x_test) == 0) & (y_test == 0))
FP = np.sum((model.predict(x_test) == 1) & (y_test == 0))

precision = TP / (TP + FP)
recall = TP / (TP + TN)
f_score = 2 * precision * recall / (precision + recall)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F-score: {f_score}')