# from sktime.transformations.panel import augmenter as aug
from sktime.transformations.series import augmenter as aug
from sktime.datasets import load_basic_motions
import pandas as pd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
from scipy.stats import norm
import traceback
import pytest


SAVE_PATH = "/code/img/"
N_ITER = 5

np.random.seed(42)


def test_seq_aug_pipeline():
    """Test of the sequential augmentation pipeline."""
    pipe = aug.SeqAugPipeline(
        [
            ("invert", aug.InvertAugmenter(p=0.5)),
            ("reverse", aug.ReverseAugmenter(p=0.5)),
            (
                "white_noise",
                aug.WhiteNoiseAugmenter(
                    p=0.5,
                    param=1.0,
                    use_relative_fit=True,
                    relative_fit_stat_fun=np.std,
                    relative_fit_type="instance-wise",
                ),
            ),
        ]
    )
    # create naive panel with 20 instances and two variables and binary target
    n_vars = 2
    n_instances = 20
    X = pd.DataFrame([[pd.Series(np.linspace(-1, 1, 10))] * n_vars] * n_instances)
    y = pd.Series(np.random.rand(n_instances) > 0.5)
    pipe.fit(X, y)
    Xt = pipe.transform(X)
    assert _calc_checksum(X) == 0.0
    assert _calc_checksum(Xt) == 45.75849112232849


def _load_test_data():
    # get some multivariate panel data
    le = preprocessing.LabelEncoder()
    X_tr, y_tr = load_basic_motions(split="train", return_X_y=True)
    X_te, y_te = load_basic_motions(split="test", return_X_y=True)
    y_tr = pd.Series(le.fit(y_tr).transform(y_tr))
    y_te = pd.Series(le.fit(y_te).transform(y_te))
    return (X_tr, X_te, y_tr, y_te)


def _train_test(data, augmentator):
    X_tr, X_te, y_tr, y_te = data
    # fit augmenter object (if necessary)
    augmentator.fit(X_tr, y_tr)
    # transform new data with (fitted) augmenter
    Xt = augmentator.transform(X_te, y_te)
    # check if result seems (trivially) invalid
    return Xt


def _calc_checksum(X):
    if isinstance(X, pd.DataFrame):
        checksum = sum([sum([sum(x) for x in X[c]]) for c in X.columns])
    else:
        checksum = sum(X)
    return checksum


## Test Data
expected_checksums_data = [646.1844410000003, -278.36259900000056, 60, 60]


def test_loaded_data():
    data = _load_test_data()
    checksums = []
    for d in data:
        checksums.append(_calc_checksum(d))
    assert checksums == expected_checksums_data


## Test WhiteNoiseAugmenter
expected_shapes_white_noise = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_white_noise = [
    -2182.947306824031,
    -1098.0137330098862,
    -1140.4422249274562,
    -31370.170890339476,
    -3803.9702383398067,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_white_noise(parameter):
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.WhiteNoiseAugmenter(**para)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
        # _plot_results(augmentator, data[1], data[3], j)
    print(checksums)
    assert shapes == expected_shapes_white_noise
    assert checksums == expected_checksums_white_noise


## Test InvertAugmenter
expected_shapes_invert = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_invert = [
    -378.06333300000057,
    1010.4995409999992,
    2197.4566569999997,
    -8657.672899000001,
    7112.357442999999,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_invert(parameter):
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.InvertAugmenter(**para)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
        # _plot_results(augmentator, data[1], data[3], j)
    print(checksums)
    assert shapes == expected_shapes_invert
    assert checksums == expected_checksums_invert


expected_shapes_reverse = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_reverse = [
    -278.36259900000067,
    -278.36259900000056,
    -278.36259900000067,
    -278.36259900000243,
    -278.36259900000056,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_reverse(parameter):
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.ReverseAugmenter(**para)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
        # _plot_results(augmentator, data[1], data[3], j)
    print(checksums)
    assert shapes == expected_shapes_reverse
    assert checksums == expected_checksums_reverse


expected_shapes_scale = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_scale = [
    -31.225222476948794,
    -2669.752684202157,
    -1163.256080823444,
    -254911.97641701132,
    62569.650700660335,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_scale(parameter):
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.ScaleAugmenter(**para)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
    print(checksums)
    assert shapes == expected_shapes_scale
    assert checksums == expected_checksums_scale


expected_shapes_offset = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_offset = [
    249.7049857628042,
    -2470.76576443243,
    -1948.8738966295189,
    -491929.26983597985,
    63878.60291027138,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_offset(parameter):
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.ScaleAugmenter(**para)
        Xt = _train_test(data, augmentator)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
    print(checksums)
    assert shapes == expected_shapes_offset
    assert checksums == expected_checksums_offset


expected_shapes_drift = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_drift = [
    -142.76348938629795,
    -1925.2768903253334,
    -1878.298869053208,
    -449595.48280445003,
    62395.34555686453,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_drift(parameter):
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.ScaleAugmenter(**para)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
    print(checksums)
    assert shapes == expected_shapes_drift
    assert checksums == expected_checksums_drift


def test_random_input_all_plot():
    ## Only in "private" sample...
    X_tr, X_te, y_tr, y_te = _load_test_data()
    n_vars = X_tr.shape[1]

    # get list of all augmenters
    all_augmenter_classes = aug.BasePanelAugmenter.__subclasses__()[:1]

    # perform exhaustive stochastic testing of each individual augmenter
    err_list = []
    # loop over all augmenters
    for i, aug_cls_i in enumerate(all_augmenter_classes):
        aug_name_i = aug_cls_i.__name__
        print(aug_name_i)
        # loop over all stochastic repetitions
        for j in range(N_ITER):
            # aug.progress_bar(i * N_ITER + j + 1,
            #                    len(all_augmenter_classes) * N_ITER,
            #                    f"{aug_name_i}, rep: {j+1} of {N_ITER}")
            try:
                # initialize augmenter object
                aug_obj_i_j = aug_cls_i(**aug.get_rand_input_params(n_vars))
                # fit augmenter object (if necessary)
                aug_obj_i_j.fit(X_tr, y_tr)
                # transform new data with (fitted) augmenter
                Xt_te_i_j = aug_obj_i_j.transform(X_te, y_te)
                # check if result seems (trivially) invalid
                if X_te.shape != Xt_te_i_j.shape:
                    raise ValueError(
                        f"Augmentation result seems invalid for "
                        f"{aug_name_i} in repetition {j+1} of "
                        f"{N_ITER}."
                    )
                # plot and save exemplary results for subsequent manual
                # objective checking
                filename = f"{SAVE_PATH}test_{aug_name_i}_{j}b.png"
                _plot_results(aug_obj_i_j, X_te, y_te, filename)
                # c_sum = 0
                # for c in Xt_te_i_j.columns:
                #    values = [sum(x) for x in Xt_te_i_j[c]]
                #    values = sum(values)
                #    c_sum += values
                #    #print(values)
                # print(c_sum, sum([sum([sum(x) for x in Xt_te_i_j[c]]) for c in Xt_te_i_j.columns]))

            except Exception as e:
                err_list.append(
                    {
                        "aug_class": aug_name_i,
                        "rep_idx": j,
                        "err_msg:": e,
                        "traceback": traceback.format_exc(),
                    }
                )
    print(err_list)


def _plot_results(aug_obj, X, y, repeat_i):
    ## only private...
    aug_name = aug_obj.__class__.__name__
    filename = f"{SAVE_PATH}test_{aug_name}_{repeat_i}.png"
    aug.plot_augmentation_example(aug_obj, X, y)
    plt.savefig(filename)
    plt.close("all")


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = _load_test_data()
    X = X_tr.iloc[0, 1]

    _aug = aug.RandomSamplesAugmenter()
    Xt = _aug.fit_transform(X)
    print(type(Xt))


def test():
    wna = aug.WhiteNoiseAugmenter(scale=2)
    Xt = wna.fit_transform(X)

    wna = aug.WhiteNoiseAugmenter()
    Xt = wna.fit_transform(X)
    print(type(Xt))
    wna = aug.WhiteNoiseAugmenter(scale=np.std)
    Xt = wna.fit_transform(X)
    print(type(Xt))

    rea = aug.ReverseAugmenter()
    Xt = rea.fit_transform(X)
    print(type(Xt))
    # print(Xt)

    ina = aug.InvertAugmenter()
    Xt = ina.fit_transform(X)
    # print(X)
    print(type(Xt))

    _aug = aug.RandomSamplesAugmenter(n=20)
    Xt = _aug.fit_transform(X)
    print(type(Xt))
    _aug = aug.RandomSamplesAugmenter(without_replacement=False)
    Xt = _aug.fit_transform(X)
    print(type(Xt))
    ## test_seq_aug_pipeline()
    # test_loaded_data()
    # test_white_noise(parameter)
    # test_invert()
    # test_reverse()
    # test_scale()
    # test_offset()
    # test_drift()
    pass
