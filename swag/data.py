import numpy as np
import torch
import torchvision
import os
import random

from .camvid import CamVid

c10_classes = np.array([
    [0, 1, 2, 8, 9],
    [3, 4, 5, 6, 7]
], dtype=np.int32)

def camvid_loaders(path, batch_size, num_workers, transform_train, transform_test, 
                use_validation, val_size, shuffle_train=True, 
                joint_transform=None, ft_joint_transform=None, ft_batch_size=1, **kwargs):

    #load training and finetuning datasets
    print(path)
    train_set = CamVid(root=path, split='train', joint_transform=joint_transform, transform=transform_train, **kwargs)
    ft_train_set = CamVid(root=path, split='train', joint_transform=ft_joint_transform, transform=transform_train, **kwargs)

    val_set = CamVid(root=path, split='val', joint_transform=None, transform=transform_test, **kwargs)
    test_set = CamVid(root=path, split='test', joint_transform=None, transform=transform_test, **kwargs)

    num_classes = 11 # hard coded labels ehre
    
    return {'train': torch.utils.data.DataLoader(
                        train_set, 
                        batch_size=batch_size, 
                        shuffle=shuffle_train, 
                        num_workers=num_workers,
                        pin_memory=True
                ),
            'fine_tune': torch.utils.data.DataLoader(
                        ft_train_set, 
                        batch_size=ft_batch_size, 
                        shuffle=shuffle_train, 
                        num_workers=num_workers,
                        pin_memory=True
                ),
            'val': torch.utils.data.DataLoader(
                        val_set, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=num_workers,
                        pin_memory=True
                ),
            'test': torch.utils.data.DataLoader(
                        test_set, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=num_workers,
                        pin_memory=True
                )}, num_classes


def svhn_loaders(path, batch_size, num_workers, transform_train, transform_test, use_validation, val_size, shuffle_train=True):
    train_set = torchvision.datasets.SVHN(root=path, split='train', download = True, transform = transform_train)

    if use_validation:
        test_set = torchvision.datasets.SVHN(root=path, split='train', download = True, transform = transform_test)
        train_set.data = train_set.data[:-val_size]
        train_set.labels = train_set.labels[:-val_size]

        test_set.data = test_set.data[-val_size:]
        test_set.labels = test_set.labels[-val_size:]

    else:
        print('You are going to run models on the test set. Are you sure?')
        test_set = torchvision.datasets.SVHN(root=path, split='test', download = True, transform = transform_test)

    num_classes = 10

    return \
        {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
        }, \
        num_classes


def get_imagenette160(path, train, download, transform):
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz"
    root = os.path.expanduser(path)
    if not os.path.exists(root):
        raise ValueError("please download and extract data from {}".format(url))

    if train:
        root = os.path.join(root, "train")
    else:
        root = os.path.join(root, "val")
    ds = torchvision.datasets.ImageFolder(root, transform=transform)
    return ds


def imagenette_loaders(path, batch_size, num_workers, transform_train, transform_test,
                       use_validation, val_size=5000, shuffle_train=True, subsample=None):

    train_set = get_imagenette160(path, train=True, download=True, transform=transform_train)
    num_classes = 10
    if use_validation:
        raise NotImplementedError("No validation set in Imagenette")
    else:
        test_set = get_imagenette160(path, train=False, download=True, transform=transform_test)
        if subsample is not None:
            random.seed(1234)
            random.shuffle(train_set.samples)
            train_set.samples = train_set.samples[:subsample]

    print("Test datapoints:", len(test_set))
    target_lst = [target for (_, target) in train_set.samples]
    test_target_lst = [target for (_, target) in test_set.samples]
    for c in range(num_classes):
        print("{} objects in train in class {}".format(
            sum([int(label == c) for label in target_lst]), c))
        print("{} objects in test in class {}".format(
            sum([int(label == c) for label in test_target_lst]), c))

    return {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
           }, num_classes



def loaders(dataset, path, batch_size, num_workers, transform_train, transform_test, 
            use_validation=True, val_size=5000, split_classes=None, shuffle_train=True,
            subsample=None, **kwargs):

    if dataset == 'CamVid':
        return camvid_loaders(path, batch_size=batch_size, num_workers=num_workers, transform_train=transform_train, 
                        transform_test=transform_test, use_validation=use_validation, val_size=val_size, **kwargs)

    if dataset == 'ImageNette160':
        return imagenette_loaders(path, batch_size=batch_size, num_workers=num_workers, transform_train=transform_train,
                                  transform_test=transform_test, use_validation=use_validation, val_size=val_size,
                                  shuffle_train=shuffle_train, subsample=subsample, **kwargs)

    path = os.path.join(path, dataset.lower())
    
    ds = getattr(torchvision.datasets, dataset)            

    if dataset == 'SVHN':
        return svhn_loaders(path, batch_size, num_workers, transform_train, transform_test, use_validation, val_size)
    else:
        ds = getattr(torchvision.datasets, dataset)           

    if dataset == 'STL10':
        train_set = ds(root=path, split='train', download=True, transform=transform_train)
        num_classes = 10
        cls_mapping = np.array([0, 2, 1, 3, 4, 5, 7, 6, 8, 9])
        try:
            train_set.targets = cls_mapping[train_set.targets]
        except:
            train_set.labels = cls_mapping[train_set.labels]
    else:
        train_set = ds(root=path, train=True, download=True, transform=transform_train)
        try:
            num_classes = max(train_set.targets) + 1
        except:
            num_classes = max(train_set.train_labels) + 1

    if use_validation:
        print("Using train (" + str(len(train_set.data)-val_size) + ") + validation (" +str(val_size)+ ")")
        try:
            train_set.data = train_set.data[:-val_size]
            train_set.targets = train_set.targets[:-val_size]
        except:
            train_set.train_data = train_set.train_data[:-val_size]
            train_set.train_labels = train_set.train_labels[:-val_size]

        test_set = ds(root=path, train=True, download=True, transform=transform_test)
        test_set.train = False
        try:
            test_set.data = test_set.data[-val_size:]
            test_set.targets = test_set.targets[-val_size:]
        except:
            test_set.test_data = test_set.tets_data[-val_size:]
            test_set.test_labels = test_set.test_labels[-val_size:]
            delattr(test_set, 'train_data')
            delattr(test_set, 'train_labels')
        if subsample is not None:
            raise NotImplementedError('Subsample and validation are not supported')
    else:
        print('You are going to run models on the test set. Are you sure?')
        if dataset == 'STL10':
            test_set = ds(root=path, split='test', download=True, transform=transform_test)
            try:
                test_set.targets = cls_mapping[test_set.targets]
            except:
                test_set.labels = cls_mapping[test_set.labels]
        else:
            test_set = ds(root=path, train=False, download=True, transform=transform_test)

        if subsample is not None:
            try:
                train_set.data = train_set.data[:subsample]
                train_set.targets = train_set.targets[:subsample]
                target_lst = train_set.targets

                test_set.data = test_set.data[:subsample]
                test_set.targets = test_set.targets[:subsample]
                test_target_lst = test_set.targets
            except:
                train_set.train_data = train_set.train_data[:subsample]
                train_set.train_labels = train_set.train_labels[:subsample]
                target_lst = train_set.train_labels

                test_set.test_data = test_set.test_data[:subsample]
                test_set.test_labels = test_set.test_labels[:subsample]
                test_target_lst = test_set.test_labels

            for c in range(num_classes):
                print("{} objects in train in class {}".format(
                    sum([int(label == c) for label in target_lst]), c))
                print("{} objects in test in class {}".format(
                    sum([int(label == c) for label in test_target_lst]), c))

    if split_classes is not None:
        assert dataset == 'CIFAR10'
        assert split_classes in {0, 1}

        print('Using classes:', end='')
        print(c10_classes[split_classes])
        train_mask = np.isin(train_set.targets, c10_classes[split_classes])
        train_set.data = train_set.data[train_mask, :]
        train_set.targets = np.array(train_set.targets)[train_mask]
        train_set.targets = np.where(train_set.targets[:, None] == c10_classes[split_classes][None, :])[1].tolist()
        print('Train: %d/%d' % (train_set.data.shape[0], train_mask.size))

        test_mask = np.isin(test_set.targets, c10_classes[split_classes])
        test_set.data = test_set.data[test_mask, :]
        test_set.targets = np.array(test_set.targets)[test_mask]
        test_set.targets = np.where(test_set.targets[:, None] == c10_classes[split_classes][None, :])[1].tolist()
        print('Test: %d/%d' % (test_set.test_data.shape[0], test_mask.size))


    return \
        {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
        }, \
        num_classes


def loaders_inc(dataset, path, num_chunks, batch_size, num_workers, transform_train, transform_test, use_validation=True, val_size=5000, shuffle_train=True, seed=1):
    assert dataset in {'MNIST', 'CIFAR10', 'CIFAR100'}
    path = os.path.join(path, dataset.lower())

    ds = getattr(torchvision.datasets, dataset)

    train_set = ds(root=path, train=True, download=True, transform=transform_train)
    num_classes = int(max(train_set.train_labels)) + 1

    num_samples = (train_set.train_data.shape[0] - val_size) if use_validation else train_set.train_data.shape[0]
    train_sets = list()
    offset = 0

    random_state = np.random.RandomState(seed)
    order = random_state.permutation(train_set.train_data.shape[0])

    for i in range(num_chunks, 0, -1):
        chunk_size = (num_samples + i - 1) // i
        tmp_set = ds(root=path, train=True, download=True, transform=transform_train)
        tmp_set.train_data = tmp_set.train_data[order[offset:offset + chunk_size]]
        tmp_set.train_labels = np.array(tmp_set.train_labels)[order[offset:offset + chunk_size]]

        train_sets.append(tmp_set)
        offset += chunk_size
        num_samples -= chunk_size

    print('Using train %d chunks: %s' % (num_chunks, str([tmp_set.train_data.shape[0] for tmp_set in train_sets])))

    if use_validation:
        print('Using validation (%d)' % val_size)

        test_set = ds(root=path, train=True, download=True, transform=transform_test)
        test_set.train = False
        test_set.test_data = test_set.train_data[order[-val_size:]]
        test_set.test_labels = np.array(test_set.train_labels)[order[-val_size:]]
        delattr(test_set, 'train_data')
        delattr(test_set, 'train_labels')
    else:
        print('You are going to run models on the test set. Are you sure?')

        test_set = ds(root=path, train=False, download=True, transform=transform_test)

    return \
        {
            'train': [
                torch.utils.data.DataLoader(
                    tmp_set,
                    batch_size=batch_size,
                    shuffle=True and shuffle_train,
                    num_workers=num_workers,
                    pin_memory=True
                ) for tmp_set in train_sets
            ],
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
        }, \
        num_classes


