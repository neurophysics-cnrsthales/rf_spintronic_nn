import torch
from typing import Generator, Iterable
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import Dataset, DataLoader


def get_train_test_dataloaders(train_dataset: Dataset, test_dataset: Dataset, train_batch_size: int = 1,
                               test_batch_size: int = 1, shuffle: bool = True,
                               num_workers: int = 0, pin_memory: bool = False) -> tuple[DataLoader, DataLoader]:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle,
                                               num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle,
                                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader


def generator_dataloaders_for_selection(dataset_: Dataset, splits_: Iterable, train_batch_size: int = 1,
                                        test_batch_size: int = 1, shuffle: bool = True,
                                        num_workers=0,
                                        pin_memory=False):
    for train_ind, valid_ind in splits_:
        train_set = torch.utils.data.Subset(dataset_, train_ind)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=shuffle,
                                                   num_workers=num_workers, pin_memory=pin_memory)
        valid_set = torch.utils.data.Subset(dataset_, valid_ind)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=test_batch_size, shuffle=shuffle,
                                                   num_workers=num_workers, pin_memory=pin_memory)
        yield train_loader, valid_loader


def get_dataloaders(train_set: Dataset, test_set: Dataset, data_loaders_params: dict, selection_params: dict,
                    num_workers: int = 0, pin_memory: bool = False):
    if selection_params['method'] == '2ways_holdout':
        train_loader, test_loader = get_train_test_dataloaders(train_set, test_set, **data_loaders_params)
        train_valid_dataloaders = ((train_loader, test_loader) for i in range(selection_params['nb_repeats']))

    elif selection_params['method'] == 'repeated_stratified_kfold':
        kfolds = RepeatedStratifiedKFold(n_splits=selection_params['nb_splits'],
                                         n_repeats=selection_params['nb_repeats'],
                                         random_state=selection_params['random_state'])
        splits = kfolds.split(np.zeros(len(train_set.targets)), train_set.targets)
        train_valid_dataloaders = generator_dataloaders_for_selection(train_set, splits,
                                                                      **data_loaders_params,
                                                                      num_workers=num_workers,
                                                                      pin_memory=pin_memory)
    return train_valid_dataloaders
