# Model Card: Iris Species Classifier

## Model details

- **Model:** Decision tree classifier (`max_depth=3`, `random_state=42`)
- **Task:** Multiclass classification of Iris flowers as `setosa`, `versicolor`,
  or `virginica`.
- **Inputs:** Sepal length, sepal width, petal length, and petal width.
- **Output:** One predicted Iris species.

## Intended use

This model is intended for education and demonstration of an end-to-end machine
learning workflow, including fairness, explainability, and drift monitoring. It
may be used to classify measurements from flowers that are comparable to the
Iris data set.

It is not intended for high-stakes, safety-critical, medical, ecological, or
commercial decision-making.

## Training and evaluation data

The data contains 150 Iris samples, with four numerical flower measurements
and one species label. The data is split using a stratified 60% training and
40% test split (`random_state=42`) so that each species is represented in both
sets.

For the fairness exercise, a binary `location` column is randomly assigned to
each sample using a fixed random seed. It is a synthetic demonstration
attribute, not a real geographic location or protected characteristic.

## Performance

On the 60-sample test set, the model obtained the following overall metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.983 |
| Weighted precision | 0.984 |
| Weighted recall | 0.983 |

## Fairness assessment

Using Fairlearn `MetricFrame` with `location` as the sensitive feature produced
the following results:

| Location group | Accuracy | Weighted precision | Weighted recall |
| --- | ---: | ---: | ---: |
| 0 | 0.958 | 0.964 | 0.958 |
| 1 | 1.000 | 1.000 | 1.000 |

The largest observed difference was 0.042 for accuracy and recall, and 0.036
for precision. Because location was randomly assigned and each group is small,
these differences are expected to be sampling variation rather than evidence
of systematic bias.

## Explainability

SHAP summary plots show that petal length and petal width have the greatest
influence on the classifier's Virginica predictions. Higher values of these
features generally push predictions toward Virginica, while lower values push
them away. Sepal measurements have less influence in this model.

## Drift and production-performance assessment

To simulate production drift, `1.0` was added to petal length in the labelled
test data. A two-sample Kolmogorov-Smirnov test detected petal-length drift
(`KS statistic = 0.344`, `p = 0.000294`), while the other three features were
not flagged at the 0.05 significance level.

The model was evaluated on the unchanged test data and on the corresponding
shifted production data:

| Evaluation data | Accuracy | Weighted precision | Weighted recall |
| --- | ---: | ---: | ---: |
| Original test data | 0.983 | 0.984 | 0.983 |
| Simulated production data | 0.683 | 0.709 | 0.683 |

This controlled experiment shows that petal-length drift can materially reduce
model performance. In a real deployment, production labels may arrive later,
so input drift can be monitored immediately while performance drift must be
checked when verified labels become available.

## Known limitations

- The Iris data set is small, old, and limited to three species; results do not
  establish performance on other flower species or field-collected data.
- The location attribute is synthetic, so this is not evidence of fairness for
  real populations or geographic groups.
- Group metrics are based on a small test set and can vary with the random
  split or random location assignment.
- The model can be affected by data drift. A simulated shift in petal length
  was detected by a Kolmogorov-Smirnov test, illustrating that changed input
  distributions can reduce confidence in deployed predictions.
- SHAP values describe this model's behavior; they do not prove that a feature
  causally determines a flower's species.

## Monitoring and maintenance

Monitor incoming feature distributions against the training data, especially
petal length and petal width. Investigate statistically significant drift,
validate performance on newly labelled data, and retrain only after reviewing
data quality and model performance across relevant groups.
