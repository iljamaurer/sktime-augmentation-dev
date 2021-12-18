from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from numpy import np
from sktime.transformations.panel import augmenter as aug

"""
Paper Idea(s):

Dataset(s): UCR and UEA Archive
Augmenter: static Pipeline with random "elements", steps=[flip, reverse, scale, offset, whiteNoise, ...]
Feature-Extractors: standard Rocket and MiniRocket, as devised for uni- or multivariate datasets
Classifiers: As deviced by the authors (RidgeReg / LogReg)-Classifier.

Experiments:

Appendix 1) calculate augmentation "strength" so that test_acc_aug / test_acc_orig ~= 95% = threshold
 - p=1.0, calculate for full train test set.
 - better use secant method (search) (nearest result after e.g. 6 steps)
 - calculate for: {UCR} x {augmenters} x ({6 steps} x {5 fold cv})
 - don't touch test-sets here !!!!
 - use MiniRocket for speed and take mean value of 5-fold-CV for each search step (each time re-instance the MRocket).

Appendix 2) set up augmentation pipeline
 - chose best practice "strength" parameters and set up seq. pipeline
 - chose reasonable order of augmenters
 - don't touch test-sets here !!!!
 - set use_relative_fit = True and relative_fit_type = "fit"

Appendix 3) investigate on optimal augmentation_data_ratio (adr) and global p-value
 - Use UCR datasets
 - don't touch test-sets here !!!!
 - calculate acc for {0.5, 1, 2, 5, 10}_adr x {UCR} x {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}_p with preset pipeline

Main 1) investigate effect of augmentation on small training size
 - for UCR and UEA:
   - calculate test_acc with constant test dataset for: {0.1, 0.2, 0.5, 1} x {UCR, UEA} x {aug, no_aug}


Others:
 - test-time augmentation (TTA)
 - Question of N: dataset size, online, offline
 - PCA Augmentations


OORRR:

- first show for each single-augmenter: augmenter weigth -> {only train aug, only train aug, train-test-aug}_(test_acc/baseline_test_acc)
"""



def get_score_over_aug_weight(X, y, aug, est, scoring,
                              weight_support=None,
                              aug_strategy="train_test",
                              n_jobs=1,
                              n_cv=5):
    if weight_support is None:
        if aug._param_desc is None:  # in case the augmenter has no parameter
            weight_support = 1
        else:
            weight_support = (aug._param_desc["min"], aug._param_desc["min"])
    # calculate score for each weight through n_cv-fold CV
    results = []
    pipe = Pipeline([['augmenter', aug], ['estimator', est]])
    for weight in weight_support:
        pipe.steps
        results.append(cross_validate(pipe, X, y=y,
                                      scoring=scoring,
                                      cv=n_cv,
                                      n_jobs=n_jobs,
                                      verbose=0,
                                      pre_dispatch='2*n_jobs',
                                      return_train_score=True,
                                      return_estimator=False,
                                      error_score=np.nan))
                                      