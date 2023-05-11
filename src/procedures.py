import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
from torch import optim
import optuna
from optuna.trial import TrialState
from optuna.trial import Trial
from sklearn.metrics import confusion_matrix

from utils import save_config, update_config, show_dataset_counts, write_dict, write_csv_from_dict
from datasets import get_training_test_datasets
from data_loaders import get_dataloaders, get_train_test_dataloaders
from networks import instanciate_model
from visualization import plot_evaluation_metric, show_confusion_matrix, average_confusion_matrix
from utils import confusion_matrix


def get_trial_params(trial: Trial, model_name: str, nb_hidden_layers=1) -> tuple[dict, dict]:
    """Create two dictionaries containing the right hyperparameters for a given model.

    Args:
        trial (object): A Trial instance representing a process of evaluating an objective function
        model_name (str): A string containing the name of the input model
        nb_hidden_layers (int): The number of hidden layers corresponds to the number of hyperparameters per category.

    Returns:
        Two dictionaries that contains the hyperparameters associated to the input model.
    """
    if model_name == "MLP":
        hyper_params = {
            'lr': trial.suggest_float("lr", 1e-4, 1, log=True),
        }
        model_hyper_params = {}
    elif model_name == "spinMLP":
        hyper_params = {
            'lr': trial.suggest_float("lr", 5e-7, 1e-4, log=True),
        }
        model_hyper_params = {
            'add_voltage_bias': [
                trial.suggest_float("_".join(["add_voltage_bias", f"layer{i}"]), 1e-5, 1e-1, log=True)
                for i in range(nb_hidden_layers)],
            'voltage_to_current_factors': [
                trial.suggest_float("_".join(["voltage_to_current_factors", f"layer{i}"]), 1e-2, 5, log=False)
                for i in range(nb_hidden_layers)],
        }
    return hyper_params, model_hyper_params


def create_study(trial_objective: str):
    """Create an optuna study with the right direction of optimization for the desired objective function (either the
    loss, the error or the accuracy).

    Args:
        trial_objective (str): A string containing the name of the desired objective function (either the loss, the
        error or the accuracy).

    Returns:
        Optuna study
    """
    if trial_objective == "loss" or trial_objective == "error":
        study = optuna.create_study(direction="minimize")
    elif trial_objective == "accuracy":
        study = optuna.create_study(direction="maximize")
    else:
        raise ValueError(f"Trial objective {trial_objective} does not exist. Please choose between loss, error or "
                         f"accuracy")
    return study


def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])


def model_selection_objective(trial, model, model_trainer, selection_params, data_loaders_params,
                              train_test_datasets, trial_objective, device, torch_seed, verbose):
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    train_valid_dataloaders = get_dataloaders(train_test_datasets[0], train_test_datasets[1],
                                              data_loaders_params, selection_params)
    hyper_params, model_hyper_params = get_trial_params(trial, model.name, model.nb_hidden_layers)
    print(hyper_params, model_hyper_params)

    models = [model]
    if model_trainer is not None:
        models.append(model_trainer)

    criterion = nn.CrossEntropyLoss()  # Loss function

    valid_losses = []
    valid_accuracies = []
    valid_errors = []

    for it, (train_loader, valid_loader) in enumerate(train_valid_dataloaders):
        for m in models:
            m.reset_parameters(**model_hyper_params)

        optimizer = optim.Adam(m.parameters(), lr=hyper_params['lr'])

        train_metrics, valid_metrics = train_eval_loop(model, train_loader, valid_loader,
                                                       criterion, optimizer,
                                                       nb_epochs=selection_params["epochs"],
                                                       trial=trial,
                                                       trial_iter_nb=it,
                                                       device=device,
                                                       verbose=verbose,
                                                       trial_objective=trial_objective,
                                                       model_trainer=model_trainer)

        valid_losses.append(valid_metrics['loss'][-1])
        valid_accuracies.append(valid_metrics['accuracy'][-1])
        valid_errors.append(valid_metrics['error'][-1])

    mean_valid_loss = np.mean(valid_losses)
    mean_valid_accuracy = np.mean(valid_accuracies)
    mean_valid_error = np.mean(valid_errors)

    trial.set_user_attr(key="best_model", value=model)

    objective = {"loss": mean_valid_loss,
                 "accuracy": mean_valid_accuracy,
                 "error": mean_valid_error}

    return objective[trial_objective]


def model_selection(config, save_dir=os.path.join('..', 'results'), device=torch.device('cpu'),
                    with_nonidealities=False,
                    verbose=False):
    torch_seed = config["torch_seed"]
    dataset_params = config["dataset_params"]
    data_loaders_params = config["data_loaders_params"]
    model_name = config["model"]
    hidden_size = config["hidden_size"]
    selection_params = config["model_selection_params"]
    nb_trials = selection_params["nb_trials"]
    trial_objective = selection_params["trial_objective"]

    train_test_datasets = get_training_test_datasets(**dataset_params)
    print('Training dataset (from default splitting)')
    show_dataset_counts(train_test_datasets[0])
    print('Test dataset (from default splitting)')
    show_dataset_counts(train_test_datasets[1])

    inp_features = train_test_datasets[0].nb_frequencies
    out_features = len(train_test_datasets[0].classes)
    model_params = dict(network_size=tuple([inp_features] + hidden_size + [out_features]))

    if model_name == "spinMLP":
        model_params.update(**config["physical_model_params"])
        model_params.update({"input_freq": train_test_datasets[0].frequencies})

    if model_name == "spinMLP" and with_nonidealities:
        model_params["is_with_nonidealities"] = False
        model_trainer = instanciate_model(model_name, model_params, device=device)
        model_params["is_with_nonidealities"] = True
        model = instanciate_model(model_name, model_params, device=device)
    else:
        model_trainer = None
        model = instanciate_model(model_name, model_params, device=device)

    # Trick to pass several arguments to study.optimize argument callable whose accept only the trial as argument
    def model_select_objective(trial_):
        return model_selection_objective(trial_, model, model_trainer, selection_params, data_loaders_params,
                                         train_test_datasets, trial_objective, device, torch_seed, verbose=verbose)

    study = create_study(trial_objective)
    study.optimize(model_select_objective, n_trials=nb_trials, callbacks=[callback])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    dict_best_hyperparams = trial.params

    # If we consider two hidden layer we would have two voltage biases: add_voltage_bias_layer0 and
    # add_voltage_bias_layer1. However, the Class spinMLP expect a list for add_voltage_bias opt argument. This is why
    # we need to do the following manipulation:
    if model_name == "spinMLP":
        phys_hyperparams_prefix = ["add_voltage_bias", "voltage_to_current_factors"]
        for k in phys_hyperparams_prefix:
            dict_best_hyperparams[k] = [value for key, value in dict_best_hyperparams.items()
                                        if key.startswith(k)]
            keys = [key for key in dict_best_hyperparams.keys() if
                    key.startswith(k) and key not in phys_hyperparams_prefix]
            for key in keys:
                dict_best_hyperparams.pop(key)
    # Here we are modifying the config with the selected hyperparameters
    update_config(config, dict_best_hyperparams)
    save_config(config, save_dir=save_dir)
    torch.save(study.user_attrs["best_model"], os.path.join(save_dir, "selected_model.pt"))

    trial_df = study.trials_dataframe()
    trial_df = trial_df.rename(columns={'value': trial_objective})
    trial_df.to_csv(os.path.join(save_dir, 'trials_dataframe.csv'))
    plt.close('all')
    return config


def train_one_epoch(model_trainer=None):
    if model_trainer is not None:
        return train_model_trainer_one_epoch
    else:
        return train_model_one_epoch


def train_model_one_epoch(model, train_loader, criterion, optimizer, train_loss, train_correct, device=None,
                          model_trainer=None) -> tuple[float, float]:
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        prediction = output.argmax(dim=1)
        train_correct += prediction.eq(target.view_as(prediction)).sum().item()
    return train_loss / len(train_loader), train_correct


def train_model_trainer_one_epoch(model, train_loader, criterion, optimizer, train_loss, train_correct,
                                  device=None, model_trainer=None) -> tuple[float, float]:
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output_model_trainer = model_trainer(data)
        with torch.no_grad():
            output_model = model(data)
            output_model_trainer.data = output_model.data.detach()
        loss = criterion(output_model_trainer, target)
        loss.backward()
        optimizer.step()
        # Replace the weights of the trained model by the weights of the model trainer
        for i in range(model.nb_layers):
            model.layers[i].weight.data, model.layers[i].bias.data = \
                model_trainer.layers[i].weight.data, model_trainer.layers[i].bias.data
        train_loss += loss.item()
        prediction = output_model.argmax(dim=1)
        train_correct += prediction.eq(target.view_as(prediction)).sum().item()
    return train_loss / len(train_loader), train_correct


def train_eval_loop(model, train_loader, valid_loader, criterion, optimizer, device=None,
                    nb_epochs=1, verbose=False, trial=None, trial_iter_nb=None, trial_objective="accuracy",
                    model_trainer=None):
    r"""Training procedure for the neural networks.

    Attributes:
        model (nn.Module):
            Pytorch model of nn.Module objects such as fully connected layers
        train_loader:
            Pytorch DataLoader object associated to the training set.
        valid_loader:
            Pytorch DataLoader object associated to the validation set.
        criterion:
            Pytorch criterion (loss function).
        optimizer:
            Pytorch optimizer such as SGD and Adam.
        device:
            Pytorch torch.device object.
        nb_epochs (int):
            Number of time the whole input dataloaders are used during the training session.
        trial:
            Optuna trial object.
        trial_iter_nb (int):
            The index associated to the number of time the training procedure is done.
        verbose (bool):
            If true, print extra information about the training procedure such as the loss and the accuracy.
        trial_objective:
            The desired objective function for the optimization procedure.
    Returns:
        A tuple containing two dictionaries, each one containing the metrics associated to the training and
        the validation procedure respectively.
    """
    train_losses = []
    train_accuracies = []
    train_errors = []

    valid_losses = []
    valid_accuracies = []
    valid_errors = []

    length_training_set = len(train_loader.sampler)
    length_valid_set = len(valid_loader.sampler)

    train_model = train_one_epoch(model_trainer)

    for epoch in range(nb_epochs):
        train_loss = 0.0
        train_correct = 0.
        valid_loss = 0.0
        valid_correct = 0.

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        if model_trainer is not None:
            model_trainer.train()
        train_loss, train_correct = train_model(model, train_loader, criterion, optimizer, train_loss, train_correct,
                                                device=device, model_trainer=model_trainer)

        train_accuracy = 100 * train_correct / length_training_set
        train_error = 100 - train_accuracy

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_errors.append(train_error)

        ######################
        # validate the model #
        ######################

        model.eval()  # prep model for evaluation
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()
                prediction = output.argmax(dim=1)
                valid_correct += prediction.eq(target.view_as(prediction)).sum().item()

        valid_loss = valid_loss / len(valid_loader)
        valid_accuracy = 100 * valid_correct / length_valid_set
        valid_error = 100 - valid_accuracy

        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        valid_errors.append(valid_error)

        if verbose:
            print(f'Epoch: {epoch + 1} \tTraining Loss: {train_loss:.6f} \tTraining Accuracy: {train_accuracy:.3f}% \t'
                  f'Validation Loss: {valid_loss:.6f} \tValidation Accuracy: {valid_accuracy:.3f}%')

        if trial:
            if trial_objective == 'accuracy':
                trial.report(valid_accuracy, trial_iter_nb * nb_epochs + epoch)
            elif trial_objective == 'loss':
                trial.report(valid_loss, trial_iter_nb * nb_epochs + epoch)
            elif trial_objective == 'error':
                trial.report(valid_error, trial_iter_nb * nb_epochs + epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # compute confusion matrix for the last epoch
    conf_matrix = confusion_matrix(target.cpu().numpy(), prediction.cpu().numpy())

    train_metrics = {'loss': train_losses,
                     'accuracy': train_accuracies,
                     'error': train_errors}
    valid_metrics = {'loss': valid_losses,
                     'accuracy': valid_accuracies,
                     'error': valid_errors,
                     'confusion matrix': conf_matrix}

    return train_metrics, valid_metrics


def model_train_eval(config, save_dir=os.path.join('..', 'results'), device=torch.device('cpu'),
                     with_nonidealities=False,
                     verbose=False):
    torch_seed = config["torch_seed"]
    dataset_params = config["dataset_params"]
    data_loaders_params = config["data_loaders_params"]
    model_name = config["model"]
    hidden_size = config["hidden_size"]
    evaluation_params = config["model_evaluation_params"]
    nb_epochs = evaluation_params["epochs"]
    epochs = torch.arange(evaluation_params["epochs"])
    nb_repeats = evaluation_params["nb_repeats"]

    train_set, test_set = get_training_test_datasets(**dataset_params)
    classes = train_set.classes
    print('Training dataset (from default splitting)')
    show_dataset_counts(train_set)
    print('Test dataset (from default splitting)')
    show_dataset_counts(test_set)

    inp_features = train_set.nb_frequencies
    out_features = len(train_set.classes)
    model_params = dict(network_size=tuple([inp_features] + hidden_size + [out_features]))

    hyper_params = {
        "lr": config["lr"],
    }

    if model_name == "spinMLP":
        model_params.update(**config["physical_model_params"])
        model_params.update({"input_freq": train_set.frequencies})
        hyper_params.update({"add_voltage_bias": config["physical_model_params"]["add_voltage_bias"],
                             "voltage_to_current_factors": config["physical_model_params"]["voltage_to_current_factors"]
                             })

    # Setting seed for dataloaders and initialization of learning parameters
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    train_loader, test_loader = get_train_test_dataloaders(train_set, test_set, **data_loaders_params)

    criterion = nn.CrossEntropyLoss()

    if model_name == "spinMLP" and with_nonidealities:
        model_params["is_with_nonidealities"] = False
        model_trainer = instanciate_model(model_name, model_params, device=device)
        model_params["is_with_nonidealities"] = True
        model = instanciate_model(model_name, model_params, device=device)
    else:
        model_trainer = None
        model = instanciate_model(model_name, model_params, device=device)

    model_hyper_params = {key: value for key, value in hyper_params.items() if key not in ['lr', 'nb_epochs']}

    models = [model]
    if model_trainer is not None:
        models.append(model_trainer)

    train_losses_all_rep = []
    train_accuracies_all_rep = []
    train_errors_all_rep = []

    test_losses_all_rep = []
    test_accuracies_all_rep = []
    test_errors_all_rep = []
    test_conf_matrices = []

    for it in tqdm(range(nb_repeats)):
        for m in models:
            m.reset_parameters(**model_hyper_params)

        optimizer = optim.Adam(m.parameters(), lr=hyper_params['lr'])

        train_metrics_one_rep, test_metrics_one_rep = train_eval_loop(model, train_loader, test_loader, criterion,
                                                                      optimizer,
                                                                      device=device,
                                                                      nb_epochs=nb_epochs,
                                                                      verbose=verbose,
                                                                      model_trainer=model_trainer)

        train_losses_all_rep.append(train_metrics_one_rep['loss'])
        train_accuracies_all_rep.append(train_metrics_one_rep['accuracy'])
        train_errors_all_rep.append(train_metrics_one_rep['error'])

        test_losses_all_rep.append(test_metrics_one_rep['loss'])
        test_accuracies_all_rep.append(test_metrics_one_rep['accuracy'])
        test_errors_all_rep.append(test_metrics_one_rep['error'])
        test_conf_matrices.append(test_metrics_one_rep['confusion matrix'])

    std_train_loss, mean_train_loss = torch.std_mean(torch.tensor(train_losses_all_rep), dim=0)
    std_train_accuracy, mean_train_accuracy = torch.std_mean(torch.tensor(train_accuracies_all_rep), dim=0)
    std_train_error, mean_train_error = torch.std_mean(torch.tensor(train_errors_all_rep), dim=0)

    std_test_loss, mean_test_loss = torch.std_mean(torch.tensor(test_losses_all_rep), dim=0)
    std_test_accuracy, mean_test_accuracy = torch.std_mean(torch.tensor(test_accuracies_all_rep), dim=0)
    std_test_error, mean_test_error = torch.std_mean(torch.tensor(test_errors_all_rep), dim=0)

    train_metrics = {
        "epochs": epochs,
        "loss": mean_train_loss,
        "std_loss": std_train_loss,
        "accuracy": mean_train_accuracy,
        "std_accuracy": std_train_accuracy,
        "error": mean_train_error,
        "std_error": std_train_error
    }

    test_metrics = {
        "epochs": epochs,
        "loss": mean_test_loss,
        "std_loss": std_test_loss,
        "accuracy": mean_test_accuracy,
        "std_accuracy": std_test_accuracy,
        "error": mean_test_error,
        "std_error": std_test_error,
        "confusion matrices": test_conf_matrices
    }

    torch.save(model, os.path.join(save_dir, 'model.pt'))
    save_config(config, save_dir=save_dir)

    average_conf_matrix = average_confusion_matrix(test_metrics['confusion matrices'])
    train_test_accuracies = {"epochs": epochs,
                             "train_acc": train_metrics["accuracy"],
                             "train_acc_std": train_metrics["std_accuracy"],
                             "test_acc": test_metrics["accuracy"],
                             "test_acc_std": test_metrics["std_accuracy"]
                             }

    write_dict(train_test_accuracies, os.path.join(save_dir, 'train_test_accuracies.json'))
    write_csv_from_dict(train_test_accuracies, os.path.join(save_dir, 'train_test_accuracies'))

    plot_evaluation_metric(train_metrics, test_metrics, 'loss',
                           std=(train_metrics['std_loss'], test_metrics['std_loss']),
                           save_dir=save_dir,
                           nb_epochs=nb_epochs)
    plot_evaluation_metric(train_metrics, test_metrics, 'accuracy',
                           std=(train_metrics['std_accuracy'], test_metrics['std_accuracy']),
                           save_dir=save_dir,
                           nb_epochs=nb_epochs)
    plot_evaluation_metric(train_metrics, test_metrics, 'error',
                           std=(train_metrics['std_error'], test_metrics['std_error']),
                           save_dir=save_dir,
                           nb_epochs=nb_epochs)
    show_confusion_matrix(average_conf_matrix, classes,
                          title=f'Confusion Matrix: mean Acc. {test_metrics["accuracy"][-1]:.2f} %',
                          save_dir=save_dir)


def model_testing(config, model, device=torch.device('cpu'), save_dir='./results/evaluation', verbose=False):
    torch_seed = config["torch_seed"]
    dataset_params = config["dataset_params"]
    data_loaders_params = config["data_loaders_params"]
    correct = 0.

    train_dataset, test_dataset = get_training_test_datasets(**dataset_params)
    print('Test dataset (from default splitting)')
    show_dataset_counts(test_dataset)
    classes = test_dataset.classes

    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    _, test_loader = get_train_test_dataloaders(train_dataset, test_dataset, **data_loaders_params)

    total_nb_examples = len(test_loader.sampler)

    model.eval()  # prepare model for testing
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1)  # get the index of the max log-probability
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_accuracy = 100 * correct / total_nb_examples
    test_error = 100 - test_accuracy
    conf_matrix = confusion_matrix(target.cpu().numpy(), prediction.cpu().numpy())

    if verbose:
        print(f'Test Accuracy: {test_accuracy:.3f} ({correct:.0f}/{total_nb_examples:.0f})')
        print(f'Test Error: {test_error:.3f} ({total_nb_examples - correct:.0f}/{total_nb_examples:.0f})\n')

    test_metrics = {'accuracy': test_accuracy,
                    'error': test_error,
                    'confusion matrix': conf_matrix}

    for key, val in test_metrics.items():
        print(f'{key}:\n{val}')

    show_confusion_matrix(conf_matrix, classes,
                          title=f'Confusion Matrix: Total Accuracy {test_metrics["accuracy"]:.2f} %',
                          save_dir=save_dir,
                          normalized=False)
