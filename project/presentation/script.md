# Presentation Script — Cambodia Rice Yield Prediction

---

## Slide 1 — Title

"Hi everyone, I'm TE Tikea. My project is on predicting rice yield in Cambodia using machine learning — specifically comparing three models and seeing which one actually holds up on real data."

---

## Slide 2 — Outline

"Here's what I'll go through. Background on the problem, the data I used, the three models, the results — which are interesting for a couple of reasons — and then a quick wrap-up."

---

## Slide 3 — Introduction

"I picked this topic partly because Cambodia rice data is publicly available and partly because it qualified for bonus points. But it's also a real problem — rice covers about 2 million hectares in Cambodia and any major yield drop has direct consequences for food supply and rural income. I set up two tasks: one regression task to predict the actual yield in kg per hectare, and one classification task to predict whether a year is a high or low yield year compared to the historical median."

---

## Slide 4 — Dataset

"Three data sources merged on year. FAOSTAT for the rice numbers — area harvested, production, and yield — and the World Bank climate portal for annual temperature and precipitation. After merging, I have 35 rows, 1990 to 2024. That's small. 28 for training, 7 for testing. The classification label is just above or below the median yield, which is around 2621 kg per hectare. I'll come back to why the small test set matters when we look at the results."

---

## Slide 5 — Method 1: Linear Regression

"First model is Linear Regression from scikit-learn — ordinary least squares, four features in, yield out. No hyperparameters, nothing to tune. With this dataset size, I wanted the simplest possible regression baseline before introducing anything more complex."

---

## Slide 6 — Method 2: Logistic Regression

"For classification, Logistic Regression — same linear combination but with a sigmoid at the end. Threshold at 0.5 gives the binary label. Also scikit-learn. It's the obvious starting point for a binary classification task and makes the comparison to the neural network straightforward."

---

## Slide 7 — Method 3: Neural Network

"The neural network is in PyTorch, following the same pattern from class. Architecture is 4 → 8 → 1. I trained two versions: regression uses MSE loss with no final activation, classification uses BCE loss with Sigmoid. Adam optimizer, learning rate 0.1, 500 epochs. I kept it small — on 35 samples a bigger network would just memorize the training set."

---

## Slide 8 — Results: Regression

"Linear Regression got RMSE of 335 kg per hectare. The neural network was around 529. Simpler model won. Both have negative R-squared, which sounds bad — and it is — but with only 7 test samples the metric is unstable. The NN clearly fits the training data and doesn't hold up on the test set."

---

## Slide 9 — Results: Classification

"Both models got perfect scores on classification — precision, recall, F1 all 1.0. This is actually not that surprising once you look at the data. Cambodia's yield has been going up almost every year since 1990. So all the low-yield years are from the early 90s and all the high-yield years are recent. Any model will separate those easily."

---

## Slide 10 — Results: Comparison Plot

"This plot puts it all together. Left side is regression — Linear Regression clearly beats the NN on RMSE. Right side is classification — both models max out. The story is consistent with what I described."

---

## Slide 11 — Conclusion

"To summarize: on regression, the linear model beats the neural network because 35 samples is not enough for the NN to generalize. On classification, both models do perfectly, but that says more about the data than the models. The main thing holding this project back is the dataset size. With province-level data there would be hundreds of rows instead of 35, which would make the regression results much more meaningful. That's it from me, thanks."
