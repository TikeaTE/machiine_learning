# Cambodia Rice Yield Prediction

ECAM Machine Learning project — comparative study of Linear Regression, Logistic Regression, and a shallow Neural Network on Cambodia annual rice statistics (1990–2024).

## Layout

```
project/
├── data/                 raw datasets + merged CSV
├── docs/                 assignment description and rubrics
├── outputs/plots/        generated figures
├── presentation/         beamer slides + speaker script
├── report/               IEEEtran report
└── src/                  source code
```

## Run

From this `project/` directory:

```bash
pip install -r requirements.txt
python src/main.py
```

This will (1) merge the three raw CSVs into `data/cambodia_merged.csv`, (2) generate the EDA figure, (3) train all three models and produce all comparison plots in `outputs/plots/`.

Individual steps are also runnable:

```bash
python src/prepare_data.py    # merge + save CSV
python src/eda.py             # correlation + yield-over-time plot
python src/regression.py      # linear regression + 5-fold CV
python src/classification.py  # logistic regression + 5-fold stratified CV
python src/neural_network.py  # PyTorch NN + CV
python src/compare.py         # all of the above + comparison plots
```

## Reproducibility

Random seeds are fixed (`numpy` and `torch` use seed 42; sklearn `KFold` / `StratifiedKFold` use `random_state=42`). Running `python src/main.py` twice produces identical outputs.

## Build the report / slides

```bash
cd report && pdflatex report.tex && pdflatex report.tex
cd ../presentation && pdflatex presentation.tex && pdflatex presentation.tex
```
Two passes are required for the bibliography / table-of-contents references to resolve.
