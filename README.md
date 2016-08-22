## Machine learning library
A list of supervised learning algorithm implemented by myself for text classification.

## Prerequisites
* [scikit-learn](http://scikit-learn.org/) (>=0.17, for comparison and textual data processing)

## Settings
10-fold cross-validation is employed to measure the generalization performance of the trained model.

## Experiment results
Dataset | `bernoulli NB` | `bernoulli NB (sklearn)` | `multinomial NB` | `multinomial NB (sklearn)` | `gaussian NB` | `gaussian NB (sklearn)` 
--- | --- | --- | --- | ---
MR | 75.76% | 77.78% | 74.15% | 74.15% | 66.58% | 66.58%
//Todo
