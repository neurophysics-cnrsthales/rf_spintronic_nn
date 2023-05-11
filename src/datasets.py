import os

import h5py
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.datasets import load_digits

from freq_distributions import freq_distribution


def get_training_test_datasets(dataset_path=r'..\datasets', input_freq_min=0., input_freq_max=1.0,
                               name="drones_signals",
                               type_freq_distrib="linear"):
    if name == "MNIST":
        transform = transforms.ToTensor()  # Normalization [0,1] included
        train_set = datasets.MNIST(dataset_path=dataset_path, train=True,
                                   download=True, transform=transform)
        test_set = datasets.MNIST(dataset_path=dataset_path, train=False,
                                  download=True, transform=transform)

        nb_input_frequencies = train_set.data.shape[-1] ** 2
        train_set.nb_frequencies = nb_input_frequencies
        test_set.nb_frequencies = nb_input_frequencies

        train_set.frequencies = freq_distribution(type_freq_distrib, nb_input_frequencies,
                                                  minimum=input_freq_min, maximum=input_freq_max,
                                                  scaling_factor=input_freq_min)
        test_set.frequencies = freq_distribution(type_freq_distrib, nb_input_frequencies,
                                                 minimum=input_freq_min, maximum=input_freq_max,
                                                 scaling_factor=input_freq_min)
    elif name == "drones_signals":
        train_set = BasakDroneDataset(dataset_path=dataset_path, train=True, fmin=input_freq_min, fmax=input_freq_max)

        test_set = BasakDroneDataset(dataset_path=dataset_path, train=False, fmin=input_freq_min, fmax=input_freq_max)

    return train_set, test_set


class BasakDroneDataset(Dataset):
    """Pytorch Dataset that contains the RF signals of the dataset from https://zenodo.org/record/7646236#.Y-4QbBOZOqU.

    """

    classes = ["dx4e", "dx6i", "MTx", "Nineeg", "Parrot", "q205", "S500", "tello", "WiFi", "wltoys"]

    def __init__(self, dataset_path=r'..\datasets', train=True, fmin=0.02, fmax=0.12,
                 filename='RadioSpin_D62_RF_fingerprinting.h5'):
        self.nb_classes = len(self.classes)
        self.dataset_path = dataset_path
        self.filename = filename
        self.train = train
        self.data, self.targets = self.load_data()
        self.nb_frequencies = self.data.shape[1]
        self.fmin = fmin
        self.fmax = fmax
        self.frequencies = torch.linspace(self.fmin, self.fmax, self.nb_frequencies)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        signal = self.data[idx]
        target = self.targets[idx]
        return signal, target

    def load_data(self):
        file = os.path.join(self.dataset_path, self.filename)
        with h5py.File(file, 'r') as h5f:
            if self.train:
                signals = np.squeeze(h5f['X_train'])
                targets = np.array(h5f['Y_train'][()])
            else:
                signals = np.squeeze(h5f['X_test'])
                targets = np.array(h5f['Y_test'][()])
        signals, targets = torch.FloatTensor(signals), torch.LongTensor(targets)
        return signals, targets

    def visualize_data(self, mean_signal, targets, idx):
        font_size = 18
        fig, ax = plt.subplots()
        ax.plot(mean_signal[idx])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.set_xlabel("Frequency bins", fontsize=font_size)
        if self.is_normalized:
            ax.set_ylabel("Normalized power", fontsize=font_size)
        else:
            ax.set_ylabel("Power", fontsize=font_size)
        ax.set_title(self.classes[targets[idx]], fontsize=24)
        plt.tight_layout()
        # plt.show()

    def counts(self, inds=None):
        dic = {}
        if isinstance(inds, (list, np.ndarray)):
            all_counts = torch.unique(self.targets[inds], sorted=True, return_counts=True)[1]
        else:
            all_counts = torch.unique(self.targets, sorted=True, return_counts=True)[1]
            if self.train:
                print('Training set:')
            else:
                print('Test set:')
        print(f'{"Class":<8} {"Counts"}')
        for i, counts in enumerate(all_counts):
            dic.update({self.classes[i]: counts.item()})
            print(f'{self.classes[i] + ":":<8} {counts.item()}')
        print(f'Total samples: {all_counts.sum().item()}')
        dic.update({'total': all_counts.sum().item()})
        print('')
        return dic
