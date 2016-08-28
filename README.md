## Supervised learning library
A list of supervised learning algorithm implemented by myself for text classification.

## Prerequisites
* [scikit-learn](http://scikit-learn.org/) (>=0.17, for comparison and textual data processing)
* [numpy](http://www.numpy.org/) (for list element operation)

## Settings
* `Multinomial naive Bayes`: alpha=0.01
* `Bernoulli naive Bayes`: alpha=1.0
* `Gaussian naive Bayes`: alpha=1.0
Note: 10-fold cross-validation is employed to measure the generalization performance of the trained model.

## Experiment results
#### naive Bayes model
Dataset | `bernoulli NB` | `bernoulli NB (sklearn)` | `multinomial NB` | `multinomial NB (sklearn)` | `gaussian NB` | `gaussian NB (sklearn)` 
--- | --- | --- | --- | --- | --- | ---
MR | 77.81% | 77.90% | 74.15% | 74.15% | 66.58% | 66.58%
SUBJ | 91.9% | 91.67% | 91.15% | 91.26% | 80.72% | 80.72%
CR | 79.28% | 77.20% | 76.89% | 76.89% | 55.80% | 55.80%
MPQA | 84.01% | 83.90% | 84.98% | 84.84% | 70.31% | 70.31%
//Todo
