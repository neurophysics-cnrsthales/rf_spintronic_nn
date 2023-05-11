"""
This code is used to run one of the three procedures that are model selection (including training and evaluation),
training and evaluation and evaluation.
"""
import os
import time
import argparse
from datetime import datetime

import torch
import matplotlib.pyplot as plt

from utils import load_config, override_config
from procedures import model_selection, model_train_eval, model_testing

# region ArgumentParser
parser = argparse.ArgumentParser(description="Command_lines")
parser.add_argument('--torch-seed', type=int, metavar='seed',
                    help="Fix the torch.manual_seed for reproducibility")
# dataset
parser.add_argument('--dataset_name', type=str, metavar='dn',
                    choices=["MNIST, drones_signals"],
                    help="Path of the file containing the dataset of drones' signals")
parser.add_argument('--input-freq-min', type=float, metavar='ifmin',
                    help='Min input frequency with 1 = 1 GHz')
parser.add_argument('--input-freq-max', type=float, metavar='ifmax',
                    help='Max input frequency with 1 = 1 GHz')
# Dataloaders settings
parser.add_argument('--train-batch-size', type=int, metavar='tbsz',
                    help='batch size for training')
parser.add_argument('--test-batch-size', type=int, metavar='testbsz',
                    help='batch size for testing')
parser.add_argument('--shuffle', type=int, metavar='sh',
                    help='If True, shuffle the data in the dataloaders (default: True)')
# hyper-parameter
parser.add_argument('--lr', type=float, metavar='lr',  # 1e-8
                    help='learning rate')
# MODEL
parser.add_argument('--model', type=str, metavar='model',
                    choices=["MLP, spinMLP"],
                    help='model to train')
parser.add_argument('--hidden-size', nargs='+', type=int, metavar='hsz',
                    help='List containing the number of hidden units per layer')
# Device and Material parameters
parser.add_argument('--add-voltage-bias', nargs='+', type=float, metavar='avb',
                    help='Additional bias used in the hidden layer.')
parser.add_argument('--freq-res-bounds', nargs='+', type=float, metavar='frb',
                    help='A list that contains the frequency bounds of each layers of a network.')
parser.add_argument('--freq-res-distrib', nargs='+', type=str, metavar='frd',
                    help='A list that contains the frequency distribution types of each layer of the spinMLP.')
parser.add_argument('--nb-input-resonators', type=int, metavar='NIR',
                    help='number of resonators per chains for the first layer.')
parser.add_argument('--voltage_to_current_factors', nargs='+', type=float, metavar='v2c',
                    help='The coefficient that convert the output voltage of the chains into current for the input of '
                         'the STNOs')
# Resonators
parser.add_argument('--bias-scaling', type=float, metavar='bsc',
                    help='Bias scaling factor')
parser.add_argument('--damping', type=float, metavar='dp',
                    help='Gilbert damping')
parser.add_argument('--freq-var-percentage', type=float, metavar='fvp',
                    help='Percentage of frequency variability')
parser.add_argument('--signed-connection', type=str, metavar='sc',
                    help='The way to connect electrically resonators. It can be either k or k+1. (default: k+1)')
parser.add_argument('--weight-scaling', type=float, metavar='wsc',
                    help='Weight scaling factor')
parser.add_argument('--Ith-res', type=float, metavar='ithr',
                    help='Threshold current in micro amp')

# Oscillators (Activation function)
parser.add_argument('--amp-factor', type=float, metavar='amp',
                    help='The factor of amplification of the output powers of the layer of oscillators')
parser.add_argument('--Ith-osc', type=float, metavar='itho',
                    help='Threshold current in micro ampere')
parser.add_argument('--I-clamp', type=float, metavar='ic',
                    help='Maximum current allowed to flow through oscillators')
parser.add_argument('--power-var-percentage', type=float, metavar='pvp',
                    help='Percentage of power variability')
parser.add_argument('--Q', type=float, metavar='Q',
                    help='Non-linear damping coefficient')
parser.add_argument('--R-osc', type=float, metavar='ro',
                    help='Resistance of the oscillators')
parser.add_argument('--scaling', type=float, metavar='scl',
                    help='Scaling factor used to adjust the power units from W to mu W')

# model evaluation
parser.add_argument('--eval-epochs', type=int, metavar='eve',
                    help='Number of epochs to train the model during the evaluation procedure')
parser.add_argument('--eval-nb-repeats', type=int, metavar='evr',
                    help='number of repetitions of the training session')
# Model selection
parser.add_argument('--select-epochs', type=int, metavar='sele',
                    help='number of epochs to train')
parser.add_argument('--select-method', type=str, metavar='selm',
                    choices=['2ways_holdout', 'stratified_kfold', 'repeated_stratified_kfold'],
                    help='resampling method used for the model selection')
parser.add_argument('--select-nb-repeats', type=int, metavar='selr',
                    help='Number of repetitions (default: 1)')
parser.add_argument('--select-nb-splits', type=int, metavar='sels',
                    help='Number of splits (or folds) used for the model selection. Valid for the '
                         'stratified k-folds methods')
parser.add_argument('--select-nb-trials', type=int, metavar='selt',
                    help='Number of trials for the model selection procedure')
parser.add_argument('--select-random-state', type=int, metavar='selrs',
                    help='Random state used for the choice of the splits')
parser.add_argument('--select-trial-objective', type=str, metavar='selto',
                    choices=["loss", "accuracy", "error"],
                    help='Trial objective (metric) (loss, accuracy, error)')
# Save and load arguments parser
parser.add_argument('--device', type=int, default=0, metavar='Dev', help='choice of gpu')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='If true, print additional information about training (loss, ect...)')
parser.add_argument('--config-dir', type=str, default='.', metavar='confd',
                    help='Directory where the config file is')
parser.add_argument('--config', type=str, default='./config_spinMLP.yaml', metavar='conf',
                    help='Path where the config file is')
parser.add_argument('--save-dir', type=str, default='../results', metavar='sd',
                    help='Directory to save results to')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='Saving the model after the training and evaluation procedure')
parser.add_argument('--load-model', action='store_true', default=False,
                    help='Path of the model to be loaded')
parser.add_argument('--model-path', type=str, default=r'../results/model.pt', metavar='mp',
                    help='Path of the model to be loaded')
parser.add_argument('--with-nonidealities', action='store_true', default=False,
                    help='If True, nonidealities will be added to the spinMLP')
parser.add_argument('--procedure', type=str, default='train_eval', metavar='proc',
                    choices=["select_train_eval", "train_eval", "eval"],
                    help='Procedure to be executed')
args = parser.parse_args()
# endregion

if __name__ == '__main__':
    plt.close('all')
    start = time.time()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")
    if use_cuda:
        print('The current device before initialization: {}'.format(
            torch.cuda.get_device_name(torch.cuda.current_device())))
    config = load_config(args.config)
    override_config(config, args)
    UID = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
    config["UID"] = UID
    save_dir = os.path.join(r'..', 'results', args.procedure, UID + '_' + config['model'])
    os.makedirs(save_dir, exist_ok=True)
    if args.procedure == 'select_train_eval':
        best_config = model_selection(config, device=device, save_dir=save_dir, verbose=args.verbose,
                                      with_nonidealities=args.with_nonidealities)
        model_train_eval(best_config, device=device, save_dir=save_dir, verbose=args.verbose,
                         with_nonidealities=args.with_nonidealities)
    elif args.procedure == 'train_eval':
        model_train_eval(config, device=device, save_dir=save_dir, verbose=args.verbose,
                         with_nonidealities=args.with_nonidealities)
    elif args.procedure == 'eval':
        model_path = os.path.join(os.path.dirname(args.config), 'model.pt')
        model = torch.load(model_path)
        model_testing(config, model, device=device, save_dir=save_dir, verbose=args.verbose)
    # plt.show()
    elapsed_time = time.time() - start
    print('Execution time (formatted):', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
