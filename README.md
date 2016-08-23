## Machine learning library
A list of supervised learning algorithm implemented by myself for text classification.

## Prerequisites
* [scikit-learn](http://scikit-learn.org/) (>=0.17, for comparison and textual data processing)

## Settings
10-fold cross-validation is employed to measure the generalization performance of the trained model.

## Experiment results
Dataset | `bernoulli NB` | `bernoulli NB (sklearn)` | `multinomial NB` | `multinomial NB (sklearn)` | `gaussian NB` | `gaussian NB (sklearn)` 
--- | --- | --- | --- | --- | --- | ---
MR | 77.81% | 77.90% | 74.15% | 74.15% | 66.58% | 66.58%
SUBJ | 91.9% | 91.67% | 91.15% | 91.26% | 80.72% | 80.72%
//Todo
