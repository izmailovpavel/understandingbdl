from swag import data, models, utils, losses
import numpy as np

loaders, num_classes = data.loaders(
    "CIFAR10",
    "~/datasets/",
    1000,
    4,
    None,
    None,
    use_validation=False,
    split_classes=None,
    shuffle_train=False
)

train_x = loaders['train'].dataset.data
train_y = np.array(loaders['train'].dataset.targets)

np.save("train_x", train_x)
np.save("train_y", train_y)
np.random.shuffle(train_y)
np.save("shuffled_train_y", train_y)

test_x = loaders['test'].dataset.data
test_y = np.array(loaders['test'].dataset.targets)
np.save("test_x", test_x)
np.save("test_y", test_y)
