'''
Created on 21 Nov 2017


'''
import torch
import os
from torchvision import datasets, transforms
from affine_transforms import Rotation, Zoom

from torch.utils.data import Dataset, DataLoader, TensorDataset

import glob
from PIL import Image
import settings

# Remove this
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

def load_data_subset(data_aug, batch_size,workers,dataset, data_target_dir, labels_per_class=100, valid_labels_per_class = 500):
    ## copied from GibbsNet_pytorch/load.py
    import numpy as np
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
        
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    if data_aug==1:
        print ('data aug')
        train_transform = transforms.Compose(
                                             [
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=2),
                                              transforms.ToTensor(),
                                              Rotation(15),                                            
                                              Zoom((0.85, 1.15)),
                                              transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
                                           [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        print ('no data aug')
        train_transform = transforms.Compose(
                                             [transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
                                                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

    n_labels = num_classes

    def get_sampler(labels, n=None, n_valid= None):
        # Only choose digits in n_labels
        # n = number of labels per class for training
        # n_val = number of lables per class for validation
        #print type(labels)
        #print (n_valid)
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
        # Ensure uniform distribution of labels
        np.random.shuffle(indices)

        indices_valid = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n_valid] for i in range(n_labels)])
        indices_train = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid+n] for i in range(n_labels)])
        indices_unlabelled = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:] for i in range(n_labels)])
        indices_train = torch.from_numpy(indices_train)
        indices_valid = torch.from_numpy(indices_valid)
        indices_unlabelled = torch.from_numpy(indices_unlabelled)
        sampler_train = SubsetRandomSampler(indices_train)
        sampler_valid = SubsetRandomSampler(indices_valid)
        sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
        return sampler_train, sampler_valid, sampler_unlabelled


    # Dataloaders for CIFAR 10
    train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.targets, labels_per_class, valid_labels_per_class)

    labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = train_sampler, shuffle=False, num_workers=workers, pin_memory=True)
    validation = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = valid_sampler, shuffle=False, num_workers=workers, pin_memory=True)
    unlabelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = unlabelled_sampler, shuffle=False, num_workers=workers, pin_memory=True)
    test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return labelled, validation, unlabelled, test, num_classes

def load_data_subset_unpre(data_aug, batch_size,workers,dataset, data_target_dir, labels_per_class=100, valid_labels_per_class = 500):
    ## loads the data without any preprocessing##
    import numpy as np
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    
    if data_aug==1:
        print ('data aug')
        train_transform = transforms.Compose(
                                                [transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop(32, padding=2),
                                                transforms.ToTensor(),
                                                transforms.Lambda(lambda x : x.mul(255))
                                                ])
        test_transform = transforms.Compose(
                                            [transforms.ToTensor(),
                                             transforms.Lambda(lambda x : x.mul(255))])
    else:
        print ('no data aug')
        train_transform = transforms.Compose(
                                            [transforms.ToTensor(),
                                             transforms.Lambda(lambda x : x.mul(255))
                                            ])
        test_transform = transforms.Compose(
                                            [transforms.ToTensor(),
                                             transforms.Lambda(lambda x : x.mul(255))])

    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

    n_labels = num_classes

    def get_sampler(labels, n=None, n_valid= None):
        # Only choose digits in n_labels
        # n = number of labels per class for training
        # n_val = number of lables per class for validation
        #print type(labels)
        #print (n_valid)
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
        # Ensure uniform distribution of labels
        np.random.shuffle(indices)

        indices_valid = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n_valid] for i in range(n_labels)])
        indices_train = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid+n] for i in range(n_labels)])
        indices_unlabelled = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:] for i in range(n_labels)])
        indices_train = torch.from_numpy(indices_train)
        indices_valid = torch.from_numpy(indices_valid)
        indices_unlabelled = torch.from_numpy(indices_unlabelled)
        sampler_train = SubsetRandomSampler(indices_train)
        sampler_valid = SubsetRandomSampler(indices_valid)
        sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
        return sampler_train, sampler_valid, sampler_unlabelled

    # Dataloaders for CIFAR 10
    train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.train_labels, labels_per_class, valid_labels_per_class)

    labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = train_sampler, shuffle=False, num_workers=workers, pin_memory=True)
    validation = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = valid_sampler, shuffle=False, num_workers=workers, pin_memory=True)
    unlabelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = unlabelled_sampler,shuffle=False,  num_workers=workers, pin_memory=True)
    test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return labelled, validation, unlabelled, test, num_classes


if __name__ == '__main__':
    labelled, validation, unlabelled, test, num_classes  = load_cifar10_subset(data_aug=1, batch_size=32,workers=1,dataset='cifar10', data_target_dir="/u/vermavik/data/DARC/cifar10", labels_per_class=100, valid_labels_per_class = 500)
    for (inputs, targets) in labelled:
        import pdb; pdb.set_trace()
        print (inputs)
