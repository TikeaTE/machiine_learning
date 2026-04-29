# Presentation Script — Cambodia Rice Yield Prediction
# Delivery notes are in [brackets]. Don't read those out loud.

---

## Slide 1 — Title

Hi everyone, I'm Tikea. My project is about predicting rice yield in
Cambodia using machine learning. I tried three models and compared them.

[Pause, then click.]

---

## Slide 2 — Outline

Quick outline. I'll go through the problem, the data, the three models,
the results, and a short conclusion. The results part has one weird thing
in it that I'll explain when we get there.

---

## Slide 3 — Introduction

Why rice? Rice is the main crop in Cambodia. If yields drop, food supply
and rural income take a direct hit. So it's a real problem to try to
predict.

I set up two tasks. One is regression, predicting the actual yield in
kilograms per hectare. The other is classification, predicting whether a
given year is high yield or low yield compared to the historical median.

I also evaluated everything two ways: a chronological split and 5-fold
cross-validation. The two protocols give pretty different results, which
is the weird thing I'll come back to.

---

## Slide 4 — Dataset

I merged three sources on year. FAOSTAT for rice statistics, World Bank
for temperature, and World Bank again for precipitation. After merging I
end up with 35 rows, covering 1990 to 2024.

One thing I want to flag. The raw FAOSTAT file also has a "production"
column, and I dropped it. The reason is that yield is defined as
production divided by area, so if I kept production as a feature, the
model would basically be reading the answer. So my features are area
harvested, temperature, and rainfall.

For the classification task, the threshold between high and low yield is
the median, but I compute that median only on the training set, so the
test years don't leak any information.

---

## Slide 5 — Data overview

This is the data. On the left, yield grows almost every year between
1990 and 2024. That trend matters a lot and it'll come up again. The
dashed red line is where I split train and test for the chronological
protocol.

On the right is the correlation matrix. Area harvested correlates with
yield at 0.98, which is very high. Temperature and rainfall are both
around 0.30. So at the country level the main signal really comes from
mechanisation and how much land is being farmed, not from the weather.

---

## Slide 6 — Method 1: Linear Regression

The first model is plain linear regression from scikit-learn. Ordinary
least squares. Three features in, yield out, no hyperparameters to tune.

I picked it because with only 35 rows, the simpler the model, the safer.

---

## Slide 7 — Method 2: Logistic Regression

For the classification task, logistic regression. Same linear combination
as before, but passed through a sigmoid so the output is a probability
between 0 and 1. If the probability is above 0.5, I predict high yield.
Also scikit-learn.

It's the obvious counterpart to linear regression, and it gives me
something to compare the neural network against on the same task.

---

## Slide 8 — Method 3: Neural Network

The neural network is in PyTorch. Three inputs, one hidden layer with 8
neurons and ReLU, one output.

For regression there's no activation on the output and the loss is MSE.
For classification I add a sigmoid and switch to BCE loss.

Optimiser is Adam with learning rate 0.1, 500 epochs, and I fix the seed
to 42 so the results are reproducible. I kept the network small because
on 35 rows, anything bigger would just memorise the training data.

---

## Slide 9 — Results: Regression (chronological split)

[This is the slide with the weird negative numbers. Slow down here.]

Chronological split. Train on 1990 to 2017, test on 2018 to 2024.

Linear regression gets RMSE 250, the neural network gets 453. So on this
split, linear regression is roughly twice as accurate. But both have
negative R squared, which usually looks like the model is broken.

If you check the baseline row, predicting the average of the training
years for every test year gives RMSE 1188. So linear regression is about
five times better than doing nothing. It's not a bad model.

The reason R squared still goes negative is in the box on the right. The
highest training-year yield is 3297 kg per hectare, and the lowest test
year is 3335. Every test point sits above anything the model has ever
seen, so the model is being asked to extrapolate, and R squared penalises
that very heavily. So this is really a data-shift issue, not a model
issue.

---

## Slide 10 — Results: Regression (5-fold CV)

Now the cross-validation. Instead of one fixed split I do 5 random folds,
each one a different 80/20, and I average the results.

Linear regression: R squared 0.92, plus or minus 0.05.
Neural network: 0.87, plus or minus 0.08.

So once the test points are interleaved with the training points instead
of sitting beyond them, both models work really well. Linear regression
is slightly ahead, with smaller variance. With this little data, the
simple model is the more reliable one.

---

## Slide 11 — Results: Classification

On the chronological split, both models get F1 of 1.00. That looks
perfect, but it's actually a trick of the split.

The reason is in the box on the right. All 7 test years are above the
median, because yield kept growing. So a model that just always predicts
"high" also gets F1 of 1.00. The split itself can't separate good models
from useless ones on this task.

The number to look at is the cross-validation table below. Logistic
regression gets F1 0.93, the neural network gets 0.91, and the error bars
overlap. So in practice, they tie.

---

## Slide 12 — Results: Comparison Plot

This is the chronological-split summary in one figure.

Left panel is RMSE. Linear regression has the smallest bar. Middle panel
is R squared, all negative because of the extrapolation issue I just
talked about. Right panel is classification, both at 1.00 — and again,
that's only because the test set is all positive examples, not because
the models are perfect.

So this figure is really a story about the split, more than about the
models.

---

## Slide 13 — Results: Cross-Validation

Same models, but evaluated with cross-validation, which is the comparison
I trust on this dataset.

Both panels show linear regression slightly ahead with smaller error bars
on both tasks. For a 35-row dataset, this is the result I'd lead with.

---

## Slide 14 — Neural Network Training Curve

One last figure. This is the neural network training on the regression
task, chronological split.

The blue curve is training loss, which drops to almost zero. The red
curve is test loss, which stays high. That gap is overfitting. The
network has memorised the training years and can't generalise to the
test ones. Under cross-validation, where the test set is in-distribution,
the same network behaves much better.

---

## Slide 15 — Conclusion

A few things to take away.

Under cross-validation, linear regression wins. 35 rows is just too small
for a neural network to be worth its capacity. On the chronological
split, both models look bad on regression, but that comes from the test
years being outside the training range, not from the models themselves.
On classification, the perfect scores on the chronological split are an
artefact, and under cross-validation the two models effectively tie.

The biggest fix here is more data. Province-level data would give roughly
25 times more rows. Adding fertiliser, irrigation and flood data would
also help the climate signal actually show up. And once there's enough
data, an LSTM becomes a reasonable next step.

Thanks for listening. Happy to take any questions.

---
---

# Q&A — anticipated teacher questions

Short, defendable answers. Read each one out loud once before tomorrow so
the words feel natural in your mouth.

---

**Q1. With only 35 rows, isn't this dataset too small to draw any real
conclusions?**

Honestly, yes — that's a fair criticism. 35 rows is small. That's
exactly why I report two evaluation protocols and not just one, and why I
trust the cross-validation numbers more than any single split. It's also
why I kept the neural network small. The conclusion I draw isn't "linear
regression is the best model for this problem in general" — it's "on
this dataset, the simple model is the safer one." With province-level
data the comparison would be more meaningful.

---

**Q2. Aren't you basically just predicting the time trend? Area harvested
grows over time and yield grows over time, so any model that uses area is
really just using year as a hidden variable.**

That's a real concern. Area harvested correlates with yield at 0.98, so
the model is mostly learning that trend. I didn't include year as a
feature directly, but yes, area is acting as a proxy for it. A pure
time-trend baseline would probably do quite well too. To really separate
the climate effect from the time trend I'd need either province-level
data, or detrended yield as the target. That's something I'd do in a
follow-up.

---

**Q3. Why didn't you use regularisation, like Ridge or Lasso?**

I tried plain linear regression first because it's the simplest baseline
and it was already doing very well under cross-validation, R squared 0.92.
With three features and 28 training rows, regularisation could help, but
the gain would be small. If I had more features I'd definitely add Ridge.

---

**Q4. Why 5-fold CV and not leave-one-out?**

5-fold gives me 5 estimates of the metric, which is enough to compute a
mean and standard deviation. Leave-one-out would give me 35 estimates but
each test set is a single point, so the per-fold metrics are very noisy
and you can't really compute meaningful R squared per fold. 5-fold felt
like the right balance for this size.

---

**Q5. Why did the neural network overfit if it only has 8 hidden neurons?**

Even a small network has enough capacity to memorise 28 points, especially
with 500 epochs and a learning rate of 0.1. The training curve on slide 14
shows it clearly: training loss collapses while test loss stays high. I
could have used early stopping or weight decay to fight that, but the
cleaner answer is that for this size of data, even a small network is
overparameterised. Linear regression has effectively 4 parameters; my NN
has around 40.

---

**Q6. Did you tune any hyperparameters?**

Not really. The NN settings — Adam, lr 0.1, 500 epochs, 8 hidden units —
came from the course exercises. With only 35 rows there isn't really
enough data to do a proper search without holding out yet another
validation set, which would make the test set even smaller. I treated the
hyperparameters as fixed and focused on comparing models under the same
conditions.

---

**Q7. Why use the median as the classification threshold?**

The median splits the dataset 50/50, so it gives the most balanced
classes possible. Anything else would create class imbalance that I'd
then have to deal with. I also computed the median on the training set
only, never on the full data, to avoid leakage.

---

**Q8. Why isn't temperature or rainfall more predictive?**

At the country level, climate variation gets averaged out — bad weather
in one province is offset by good weather in another. The dominant signal
in country-aggregated data is really mechanisation and intensification,
which area harvested captures. To see climate effects I'd need
province-level data where local weather actually drives local outcomes.

---

**Q9. Did you standardise the features? How?**

Yes, for the neural network. I compute the mean and standard deviation
on the training set only, and apply the same transform to the test set.
That keeps statistics from the test data from leaking into the training
process. Linear regression and logistic regression don't strictly need
standardisation since they're scale-invariant, but I do it for
consistency.

---

**Q10. R squared can't really be minus 96 — how is that even possible?**

R squared is defined as 1 minus the ratio of the model's squared error to
the variance of the target. If the model is much worse than predicting
the mean, that ratio can be huge, and R squared can go arbitrarily
negative. Minus 96 means the baseline's squared error is 97 times the
variance of the test set, which makes sense because the test years are
all clustered far above the training mean. The variance of the test set
is small, but the bias of "predict the train mean" is huge.

---

**Q11. Could a simple ARIMA or trend-line model do just as well?**

Probably yes, on regression. Yield is nearly monotonic so a linear trend
on year alone would do well under CV. The point of this project wasn't to
find the best forecaster — it was to compare three classes of model on
the same data. A time-series baseline would be a fair next addition.

---

**Q12. What does the linear regression actually learn? What are its
coefficients?**

After standardisation, area harvested is the dominant coefficient by far,
which lines up with the 0.98 correlation. Temperature and rainfall have
much smaller coefficients. So the model is essentially saying: yield
goes up when more land is being cultivated, and the climate variables
fine-tune the prediction by a small amount.

---

**Q13. Why no validation set?**

Because the dataset is too small. With 35 rows split 28/7, holding out
another fold for validation would leave maybe 20 rows for training, which
is too few. Cross-validation handles the same role — every row gets to
be in the test set once — without giving up training data. It's the
cleaner option for a dataset this size.

---

**Q14. If the chronological split is so unfair, why include it at all?**

Because it's the realistic deployment setting. If I'm forecasting yield
for 2025, I obviously can't train on 2025 data. So the chronological
split is the honest stress test. The negative R squared is informative —
it tells me that any deployed model would need to be retrained as new
years come in, and that extrapolating past the training range is risky.
That's a real finding, even if it looks bad on a slide.

---

**Q15. What would you do differently if you started over?**

Three things. One, get province-level data to multiply the row count.
Two, add a pure time-trend baseline so I can quantify how much the ML
models add over just fitting a line through year versus yield. Three, do
proper hyperparameter search on the NN with nested cross-validation, so
the comparison with linear regression is on equal footing.

---

## Survival tips

- Don't memorise the script word-for-word. Memorise the *order* of ideas.
- If you forget a number, just say "around" — "around 250 RMSE" is fine.
- If a question stumps you: "That's a good point, I'd have to look at it
  more carefully." This is a totally acceptable answer.
- The single most likely question is some version of Q1 (small dataset)
  or Q2 (just predicting the trend). Have those two ready cold.
- Breathe between slides. You have time.
