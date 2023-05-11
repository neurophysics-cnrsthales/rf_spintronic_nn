import os

import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_evaluation_metric(train_metrics: dict, test_metrics: dict, metric_name: str, std=None, save_dir=None,
                           error_bars=False, nb_epochs=2, filename_suffix='', train_min_max=None, test_min_max=None):
    epochs = np.arange(nb_epochs)
    fig, ax = plt.subplots()
    if isinstance(train_metrics, dict) and isinstance(test_metrics, dict):
        y_train = train_metrics[metric_name]
        y_test = test_metrics[metric_name]
    else:
        y_train = train_metrics
        y_test = test_metrics

    if std is not None:
        std_train = std[0]
        std_test = std[1]

        if error_bars:
            ax.errorbar(epochs, y_train, yerr=std_train, fmt='none', capsize=3, color='dodgerblue')
            ax.errorbar(epochs, y_test, yerr=std_test, fmt='none', capsize=3, color='firebrick')
        else:
            y1_train = y_train + std_train
            y2_train = y_train - std_train

            y1_test = y_test + std_test
            y2_test = y_test - std_test

            ax.fill_between(epochs, y1_train, y2_train, alpha=0.2, color='dodgerblue')
            ax.fill_between(epochs, y1_test, y2_test, alpha=0.2, color='firebrick')

            label_train = f"Training: {y_train[-1]:.2f}, std {std_train[-1]:.2f}"
            label_test = f"Test: {y_test[-1]:.2f}, std {std_test[-1]:.2f}"

    else:
        label_train = f"Training: {y_train[-1]:.2f} %"
        label_test = f"Test: {y_test[-1]:.2f} %"

    ax.plot(epochs, y_train, label=label_train, color='dodgerblue')
    ax.plot(epochs, y_test, label=label_test, color='firebrick')

    ax.set_ylabel(metric_name)
    ax.set_title("Evolution of the " + metric_name)
    ax.set_xlabel("Epochs")
    ax.grid()
    if metric_name == 'accuracy':
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim([torch.min(y_train), 100])
        ax.set_title('Evolution of the accuracy')
    elif metric_name == 'error':
        ax.set_ylabel('Error rate (%)')
        ax.set_ylim([0, torch.max(y_train)])
        ax.set_title('Evolution of the error rate')
    elif metric_name == 'loss':
        ax.set_ylabel('loss')
        ax.set_title('Evolution of the loss')
    ax.legend()

    if save_dir is not None:
        if error_bars:
            filename = "_".join(["training_test", metric_name, "error_bars"])
        else:
            filename = "_".join(["training_test", metric_name])
        fig.savefig(os.path.join(save_dir, filename + filename_suffix + '.png'), format='png')
        fig.savefig(os.path.join(save_dir, filename + filename_suffix + '.svg'), format='svg')
        fig.savefig(os.path.join(save_dir, filename + filename_suffix + '.pdf'), format='pdf')


def show_confusion_matrix(conf_matrix, classes, title='Confusion Matrix', save_dir='', normalized=True):
    if normalized:
        conf_matrix = conf_matrix / 100.
    nb_classes = conf_matrix.shape[0]
    fig, ax = plt.subplots()
    ax = plt.gca()
    norm = plt.Normalize(conf_matrix.min(), conf_matrix.max())
    im = ax.imshow(conf_matrix, norm=norm, cmap=plt.cm.Blues, interpolation='nearest')
    cmap_min, cmap_max = im.cmap(0), im.cmap(1.0)
    ax.set_xticks(np.arange(nb_classes), labels=classes)
    ax.set_yticks(np.arange(nb_classes), labels=classes)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    thresh = (conf_matrix.max() + conf_matrix.min()) / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            color = cmap_max if conf_matrix[i, j] < thresh else cmap_min
            if conf_matrix[i, j] == 0 or conf_matrix[i, j] == 100 or conf_matrix[i, j] == 1:
                ax.text(x=j, y=i, s=f"{int(conf_matrix[i, j]):1d}", va='center', ha='center', color=color)
            else:
                ax.text(x=j, y=i, s=f"{conf_matrix[i, j]:.2f}", va='center', ha='center', color=color)

    ax.set_xlabel('Predictions')
    ax.set_ylabel('Ground truth')
    ax.set_title(title)
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(conf_matrix.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(conf_matrix.shape[0] + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    if save_dir is not None:
        filename = "confusion_matrix"
        fig.savefig(os.path.join(save_dir, filename + '.png'), format='png')
        fig.savefig(os.path.join(save_dir, filename + '.svg'), format='svg')
        fig.savefig(os.path.join(save_dir, filename + '.pdf'), format='pdf')


def average_confusion_matrix(confusion_matrices):
    cumul_conf_mat = np.zeros_like(confusion_matrices[0])
    for mat in confusion_matrices:
        cumul_conf_mat += mat
    counts = cumul_conf_mat.sum(axis=1)
    cc = counts.reshape(-1, 1)
    average_conf_matrix = cumul_conf_mat / cc * 100
    return average_conf_matrix
