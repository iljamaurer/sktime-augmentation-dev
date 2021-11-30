import pandas as pd
from sktime.transformations.panel.panel_augmenter import \
    WhiteNoisePanelAugmenter, ReversePanelAugmenter
from sktime.datasets import load_basic_motions, load_unit_test
from sklearn import preprocessing
from matplotlib import pyplot as plt

# get some multivariate panel data
le = preprocessing.LabelEncoder()
X, y = load_basic_motions(split="train", return_X_y=True)
y = le.fit(y).transform(y)
y = pd.Series(y)

my_aug = WhiteNoisePanelAugmenter(p=1.0, param=5)
fig = my_aug.plot_augmentation_examples(my_aug, X, y)
plt.savefig('test1.png')

# augment with reverse and plot
my_aug = ReversePanelAugmenter(p=1.0)
fig = my_aug.plot_augmentation_examples(my_aug, X, y)
plt.savefig('test2.png')
