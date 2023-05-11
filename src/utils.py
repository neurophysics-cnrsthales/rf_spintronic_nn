import os
import csv
import json

import torch
from torch import Tensor
from torch.utils.data import Dataset
import yaml
import numpy as np
from scipy.sparse import coo_matrix


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # check_config(config)
    return config


def check_config(conf: dict):
    available_models = ["MLP", "spinMLP"]
    available_resampling_types = ["default", "custom"]
    available_resampling_methods = ["2ways_holdout", "3ways_holdout", "stratified_kfold"]
    available_trial_objectives = ["accuracy", "loss", "error"]
    if conf["resampling_params"]["resampling_type"] not in available_resampling_types:
        raise ValueError(f'Expected the resampling type {conf["resampling_params"]["resampling_type"]} to be in the '
                         f'list {available_resampling_types}')
    if conf["model"] not in available_models:
        raise ValueError(f'Expected the model name {conf["model"]} to be in the list {available_models}')

    if conf["model_evaluation_params"]["method"] not in available_resampling_methods:
        raise ValueError(
            f'Expected the resampling method {conf["model_evaluation_params"]["method"]} to be in the list '
            f'{available_resampling_methods}')
    if conf["model_selection_params"]["method"] not in available_resampling_methods:
        raise ValueError(f'Expected the resampling method {conf["model_selection_params"]["method"]} to be in the list '
                         f'{available_resampling_methods}')
    if conf["model_selection_params"]["trial_objective"] not in available_trial_objectives:
        raise ValueError(f'Expected the trial objective {conf["model_selection_params"]["trial_objective"]} to be in '
                         f'the list {available_trial_objectives}')

    # if len(conf["hidden_size"]) + 1 != len(conf["physical_model_params"]["freq_res_bounds"]): raise ValueError(
    # f'Expected the freq_res_bounds size to be equal to the hidden size + 1. It is cleaner for the ' f'simulation
    # register entry.')
    #
    # if len(conf["hidden_size"]) + 1 != len(conf["physical_model_params"]["freq_res_distrib"]): raise
    # ValueError(f'Expected the freq_res_distrib size to be equal to the hidden size + 1. It is cleaner for the' f'
    # simulation register entry.')


def override_config(config, args):
    if args.torch_seed is not None:
        config['torch_seed'] = args.torch_seed
    # Dataparam
    if args.dataset_name is not None:
        config["dataset_params"]["name"] = args.dataset_name
    if args.input_freq_min is not None:
        config["dataset_params"]["input_freq_min"] = args.input_freq_min
    if args.input_freq_max is not None:
        config["dataset_params"]["input_freq_max"] = args.input_freq_max
    # if args.dataset_filename is not None:
    #     config["dataset_params"]["is_normalized"] = args.is_normalized

    if args.train_batch_size is not None:
        config["data_loaders_params"]["train_batch_size"] = args.train_batch_size
    if args.test_batch_size is not None:
        config["data_loaders_params"]["test_batch_size"] = args.test_batch_size
    if args.shuffle is not None:
        config["data_loaders_params"]["shuffle"] = args.shuffle

    if args.lr is not None:
        config["lr"] = args.lr
    if args.model is not None:
        config["model"] = args.model
    if args.hidden_size is not None:
        config["hidden_size"] = args.hidden_size

    if args.eval_epochs is not None:
        config["model_evaluation_params"]["epochs"] = args.eval_epochs
    if args.eval_nb_repeats is not None:
        config["model_evaluation_params"]["nb_repeats"] = args.eval_nb_repeats

    if args.select_epochs is not None:
        config["model_selection_params"]["epochs"] = args.select_epochs
    if args.select_method is not None:
        config["model_selection_params"]["method"] = args.select_method
    if args.select_nb_repeats is not None:
        config["model_selection_params"]["nb_repeats"] = args.select_nb_repeats
    if args.select_nb_splits is not None:
        config["model_selection_params"]["nb_splits"] = args.select_nb_splits
    if args.select_nb_trials is not None:
        config["model_selection_params"]["nb_trials"] = args.select_nb_trials
    if args.select_random_state is not None:
        config["model_selection_params"]["random_state"] = args.select_random_state
    if args.select_trial_objective is not None:
        config["model_selection_params"]["trial_objective"] = args.select_trial_objective

    if args.add_voltage_bias is not None:
        config["physical_model_params"]["add_voltage_bias"] = args.add_voltage_bias
    if args.freq_res_bounds is not None:
        config["physical_model_params"]["freq_res_bounds"] = args.freq_res_bounds
    if args.freq_res_distrib is not None:
        config["physical_model_params"]["freq_res_distrib"] = args.freq_res_distrib
    if args.nb_input_resonators is not None:
        config["physical_model_params"]["nb_input_resonators"] = args.nb_input_resonators
    if args.voltage_to_current_factors is not None:
        config["physical_model_params"]["voltage_to_current_factors"] = args.voltage_to_current_factors

    if args.bias_scaling is not None:
        config["physical_model_params"]["resonators_params"]["bias_scaling"] = args.bias_scaling
    if args.damping is not None:
        config["physical_model_params"]["resonators_params"]["bias_scaling"] = args.damping
    if args.freq_var_percentage is not None:
        config["physical_model_params"]["resonators_params"]["freq_var_percentage"] = args.freq_var_percentage
    if args.Ith_res is not None:
        config["physical_model_params"]["resonators_params"]["Ith_res"] = args.Ith_res
    if args.signed_connection is not None:
        config["physical_model_params"]["resonators_params"]["signed_connection"] = args.signed_connection
    if args.weight_scaling is not None:
        config["physical_model_params"]["resonators_params"]["weight_scaling"] = args.weight_scaling

    if args.amp_factor is not None:
        config["physical_model_params"]["oscillators_params"]["amp_factor"] = args.amp_factor
    if args.Ith_osc is not None:
        config["physical_model_params"]["oscillators_params"]["Ith_osc"] = args.Ith_osc
    if args.I_clamp is not None:
        config["physical_model_params"]["oscillators_params"]["Iclamp"] = args.I_clamp
    if args.power_var_percentage is not None:
        config["physical_model_params"]["oscillators_params"]["power_var_percentage"] = args.power_var_percentage
    if args.Q is not None:
        config["physical_model_params"]["oscillators_params"]["Q"] = args.Q
    if args.R_osc is not None:
        config["physical_model_params"]["oscillators_params"]["R_osc"] = args.R_osc
    if args.scaling is not None:
        config["physical_model_params"]["oscillators_params"]["scaling"] = args.scaling


#
# def override_config(config, args):
#     prefixes = ['selection_', 'evaluation_']
#     excluded_sections = ['UID']
#     for section in config:
#         for key, value in vars(args).items():
#             if value is not None:
#                 if key in config[section]:
#                     config[section][remove_prefix(key, prefixes)] = value
#     return config

def update_config(config, dic) -> None:
    for key, value in dic.items():
        set_recursively(config, key, value)
        print("    {}: {}".format(key, value))


def remove_prefix(string, prefix):
    if string.startswith(prefix):
        return string[len(prefix):]
    else:
        return string


def save_config(in_config, config_name="config", save_dir='../results'):
    save_config_yaml(in_config, config_name=config_name + '.yaml', save_dir=save_dir)
    save_config_json(in_config, config_name=config_name + '.json', save_dir=save_dir)


def save_config_yaml(in_config, config_name="config_spinMLP.yaml", save_dir='../Optuna_results'):
    with open(os.path.join(save_dir, config_name), 'w') as f:
        yaml.safe_dump(in_config, f, sort_keys=False, indent=2)


def save_config_json(in_config, config_name="config.json", save_dir='../Optuna_results'):
    with open(os.path.join(save_dir, config_name), 'w') as f:
        json.dump(in_config, f, sort_keys=False, indent=2)
    # with open(os.path.join(config_dir, 'args_parameters.txt'), 'r') as f:
    #     config = json.load(f)


def get_path_from_UID(sim_dir, UID):
    config_dir = None
    for dirpath, dirnames, filenames in os.walk(sim_dir):
        for dirname in dirnames:
            if dirname.startswith(UID):
                return os.path.join(sim_dir, dirname)
    if config_dir is None:
        raise FileNotFoundError("Config dir not found")


def get_recursively(in_dict, target_key):
    if isinstance(in_dict, dict):
        if target_key in in_dict:
            return in_dict[target_key]
        for key in in_dict:
            target_value = get_recursively(in_dict[key], target_key)
            if target_value is not None:
                return target_value
    return None


def set_recursively(in_dict, lookup_key, new_val):
    if isinstance(in_dict, dict):
        for k, v in in_dict.items():
            if k == lookup_key:
                in_dict[k] = new_val
            else:
                set_recursively(v, lookup_key, new_val)
    # elif isinstance(in_dict, list):
    #     for item in in_dict:
    #         yield from item_generator(item, lookup_key)


def write_dict(in_dic: dict, path):
    dic = in_dic.copy()
    for key, value in dic.items():
        if isinstance(value, Tensor):
            dic[key] = value.tolist()
    with open(path, 'w') as f:
        json.dump(dic, f, indent=2)


def write_csv(data, header, filename):
    with open(filename, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)  # write header
        for row in data:
            csv_writer.writerow(row)  # write each row


def write_csv_from_dict(in_dic, filename):
    dic = in_dic.copy()
    field_names = dic.keys()
    for key, value in dic.items():
        if isinstance(value, Tensor):
            dic[key] = value.tolist()
    with open(filename + '.csv', 'w', newline='') as csvfile:
        # writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer = csv.writer(csvfile)
        writer.writerow(dic.keys())
        # writer.writeheader()
        writer.writerows(zip(*dic.values()))


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def counts_targets(targets, classes=None):
    print(f'{"Class":<8} {"Counts"}')
    all_counts = torch.unique(targets, sorted=True, return_counts=True)[1]
    if classes:
        for i, counts in enumerate(all_counts):
            print(f'{classes[i] + ":":<8} {counts}')
    else:
        for i, counts in enumerate(all_counts):
            print(f'{i + ":":<8} {counts}')
    print(f'Total samples: {all_counts.sum()}')
    print('')


def show_dataset_counts(dataset: Dataset):
    if isinstance(dataset, Dataset):
        counts_targets(dataset.targets, dataset.classes)


def confusion_matrix(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    else:
        sample_weight = np.asarray(sample_weight)

    n_labels = np.unique(y_true).size

    cm = coo_matrix(
        (sample_weight, (y_true, y_pred)),
        shape=(n_labels, n_labels)
    ).toarray()
    return cm
