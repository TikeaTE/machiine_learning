# Presentation Script — Cambodia Rice Yield Prediction

---

## Slide 1 — Title

"Hi everyone, I'm TE Tikea. My project is on predicting rice yield in Cambodia using machine learning — comparing three models and seeing which one actually holds up under fair evaluation."

---

## Slide 2 — Outline

"Here's what I'll go through. Background on the problem, the data, the three models, the results — which contain a small twist about how you evaluate them — and a quick wrap-up."

---

## Slide 3 — Introduction

"I picked Cambodia rice because it's a real problem with public data: rice covers about 2 million hectares and any major yield drop has direct consequences for food supply and rural income. I set up two tasks: regression to predict the actual yield in kg per hectare, and binary classification to predict whether a year is high or low yield compared to the historical median. I evaluate everything under two protocols — a chronological split and 5-fold cross-validation — because they tell quite different stories."

---

## Slide 4 — Dataset

"Three sources merged on year: FAOSTAT for rice statistics, World Bank for temperature, and World Bank again for precipitation. After merging, 35 rows from 1990 to 2024. One important note — I dropped the 'production' feature even though it was in the raw FAOSTAT file. Yield is defined as production divided by area, so including production would directly leak the target. The classification threshold is the median yield, computed only on the training portion of each split, again to avoid look-ahead bias."

---

## Slide 5 — Data overview

"This figure shows the data. On the left, yield grows almost monotonically from 1990 to 2024 — that's a key fact for everything that follows. The dashed red line is the chronological train/test split. On the right, the correlation matrix: area-harvested correlates with yield at 0.98, which dominates everything. Climate variables sit around 0.30. So at the country-aggregated level, the dominant signal is mechanisation and intensification, not weather."

---

## Slide 6 — Method 1: Linear Regression

"First model is Linear Regression from scikit-learn — ordinary least squares. Three features in, yield out. No hyperparameters. With 35 rows, the simpler the model, the better the chance of generalising."

---

## Slide 7 — Method 2: Logistic Regression

"For classification, Logistic Regression — same linear combination but with a sigmoid. Threshold at 0.5. Also scikit-learn. Obvious starting point and directly comparable to the neural network."

---

## Slide 8 — Method 3: Neural Network

"The neural network is in PyTorch. Architecture is 3 → 8 → 1, matching the three features. Two variants: regression with MSE loss and no final activation; classification with BCE loss and Sigmoid. Adam optimizer, lr 0.1, 500 epochs, seed 42 for determinism. I kept it small on purpose."

---

## Slide 9 — Results: Regression (chronological split)

"Chronological split: train on 1990-2017, test on 2018-2024. Linear Regression gets RMSE 250, Neural Network gets 453. Both have negative R-squared. Negative R-squared sounds catastrophic — but look at the baseline row: 'always predict the train mean' has RMSE 1188. So Linear Regression is about five times better than that. Why is R-squared still negative? Because the test years are all above the training maximum: train max is 3297, test minimum is 3335. Every test point sits outside the training range. That's distribution shift, and no amount of model capacity fixes it."

---

## Slide 10 — Results: Regression (5-fold CV)

"Now the cross-validation results. Five random folds. Linear Regression: R-squared 0.92 plus or minus 0.05. Neural Network: 0.87 plus or minus 0.08. So when test points are interleaved with training points instead of sitting beyond them, both models work very well, and Linear Regression edges out the NN with substantially less variance. This is exactly what the bias-variance trade-off predicts on a 35-sample dataset."

---

## Slide 11 — Results: Classification

"Classification on the chronological split: both models get F1 1.0. Looks great until you check the test set — all 7 years are above-median, because yield grew steadily over time. So 'always predict High' would also score 1.0. The chronological split simply cannot distinguish the models. The 5-fold stratified CV in the second table is the meaningful comparison: Logistic Regression at 0.93, Neural Network at 0.91, but well within one standard deviation. Effectively a tie."

---

## Slide 12 — Results: Comparison Plot

"This is the chronological-split summary. Left: RMSE — Linear Regression clearly best. Center: R-squared — all negative because of the extrapolation issue. Right: classification — both perfect, but for trivial reasons we just discussed."

---

## Slide 13 — Results: Cross-Validation

"And this is the cross-validation comparison. Both panels show Linear Regression slightly ahead with smaller error bars. This is the more credible evaluation given the small dataset."

---

## Slide 14 — Neural Network Training Curve

"For completeness — the NN regression training curve on the chronological split. Train MSE crashes to almost zero while test MSE plateaus way above it. That's textbook overfitting, and explains why the NN is worse than Linear Regression on this split. With CV folds where the test set is in-distribution, the same architecture works fine."

---

## Slide 15 — Conclusion

"To summarize: on cross-validation, the linear model wins because 35 samples is too few for a neural network to be worth its capacity. On the chronological split, both models look bad — but that's distribution shift, not the models. On classification, the chronological perfect scores are an artefact of the split; CV gives the real comparison and it's effectively a tie. The biggest improvement available is more data — province-level annual yields would give roughly 25 times more rows. Thanks."
