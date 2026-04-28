# Machine Learning Project — Claude Guidelines

## Project Overview
Cambodia Rice Yield Prediction using ML models. Goal: compare Linear Regression,
Logistic Regression, and a Neural Network on Cambodia agricultural data.

## Folder Structure
```
machine_learning/
├── course/
│   ├── lessons/        ← lesson PDFs
│   ├── exercises/      ← student + teacher notebooks
│   └── data/           ← exercise datasets
└── project/
    ├── data/           ← Cambodia datasets (source of truth)
    ├── docs/           ← project_description.pdf, report_template.docx
    ├── src/            ← project code goes here
    └── outputs/plots/  ← saved figures
```

## Project Datasets
| File | Description |
|------|-------------|
| `project/data/cambodia_rice_faostat.csv` | Rice area harvested, yield, production 1990–2024 (FAOSTAT) |
| `project/data/cambodia_rainfall_annual.csv` | Annual precipitation (mm) 1901–2024 |
| `project/data/cambodia_temperature_annual.csv` | Annual mean temperature (°C) 1901–2024 |

After merging on year (1990–2024): **35 rows, 4 features, 1 target**
- Features: `area_harvested`, `production`, `avg_temp`, `rainfall`
- Target: `yield_kg_ha` (regression) / `high_yield` (classification, above/below median)

## Models to Build
| File | Model | Task | Metrics |
|------|-------|------|---------|
| `src/regression.py` | Linear Regression (sklearn) | Predict yield (continuous) | MSE, RMSE, R² |
| `src/classification.py` | Logistic Regression (sklearn) | Predict High/Low yield | Precision, Recall, F1 |
| `src/neural_network.py` | PyTorch NN | Both tasks | All above |
| `src/compare.py` | — | Side-by-side comparison plots | — |

## Coding Style (match teacher's examples exactly)

### PyTorch NN structure
```python
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()   # for classification; remove for regression
        )

    def forward(self, x):
        return self.layers(x).flatten()
```

### Training loop
```python
model = MyModel()
opt = optim.Adam(model.parameters(), lr=0.1)
cost_func = nn.BCELoss()   # or nn.MSELoss() for regression

for i in range(num_epochs):
    output = model(tx)
    loss = cost_func(output, ty)
    loss.backward()
    opt.step()
    opt.zero_grad()
```

### Evaluation function (teacher's pattern)
```python
def compute_PRF(z, y):
    TP = np.sum((z == 1) & (y == 1))
    FN = np.sum((z == 0) & (y == 1))
    FP = np.sum((z == 1) & (y == 0))
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f_score   = 2 * precision * recall / (precision + recall)
    return precision, recall, f_score
```

### Feature scaling (always apply before training)
```python
x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
```

## Key Rules
- Use `sklearn` for baseline Linear/Logistic Regression
- Use `PyTorch nn.Module` (NOT raw numpy) for the Neural Network
- Use `Adam` optimizer, NOT plain SGD
- Use `BCELoss` for classification, `MSELoss` for regression
- Train/test split: 80/20 (28 train, 7 test) — note small dataset in comments
- High/Low yield threshold: median yield across all years
- NN architecture for this dataset: 4 → 8 → 1 (small, avoids overfitting on 35 rows)
- All project code goes in `project/src/`
- Save all plots to `project/outputs/plots/`
