# NYTimes Index Prediction Experiment
In this experiment we seek to test the broad hypothesis:  *News headlines contain information that predicts broad market movements.*

We'll do so by training fine-tuning an LLM with a classifer-head on data of the form `{input:Headline, Index-Return-Measure}`.

We'll start with an *Index Return Measure* that we'll call 3 bucket Daily Return that we'll compute as  follows:
1. For the given headline publication date (pub_date), choose the return period pub_date + 1 BD to pub_date + 2 BD.
2. Compute the Z-Score of the daily return over this period.
3. Classify that Z-Score using the follwowing bucketing rule:
  * zscore < k1: 0
  * k1 <= zscore < k2: 1
  * zscore >= k2: 2

...where we choose k1 and k2 to be reasonably equal-weighted to ease training.

## Trial-1 Protocol

1. Collect NYT headlines data from their archives API over the period: 2019-01-01..2024-01-03.
2. Collect ETF prices for SPY over the same period.
3. Compute a train- and test-dataset from these headlines.
4. Perform a fine-tuning of BERT+classifer on the training set.
5. Measure the model's predictive power on the test-dataset.


# Practical Details

## Building a Wheel
The `pyproject.yaml` is the blueprint.  To build the wheel:

```
python3 -m pip install --upgrade build
python3 -m build
```

For each new build I tick the version and tag:
1. Edit the version in `pyproject.yaml`
2. `git tag 'nyt_expt_24_0.0.2'`
3. `git push origin --tags`






# References
* [Colaboratory Launch URL](https://research.google.com/colaboratory/)
* [PyPA: Packaging Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)







