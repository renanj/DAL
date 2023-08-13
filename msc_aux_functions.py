import pandas as pd
import numpy as np
import copy
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchvision import datasets
from PIL import Image
import torch
import torch.optim as optim
from torch.autograd import Variable
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import time
import math
import random
import os
import pickle
import json


from numpy.linalg import cond
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import sparse as sp
from scipy.linalg import lstsq
from scipy.linalg import solve
from scipy.optimize import nnls

from distil.active_learning_strategies.badge import BADGE
from distil.active_learning_strategies.glister import GLISTER
from distil.active_learning_strategies.margin_sampling import MarginSampling
from distil.active_learning_strategies.entropy_sampling import EntropySampling
from distil.active_learning_strategies.random_sampling import RandomSampling
from distil.active_learning_strategies.gradmatch_active import GradMatchActive
from distil.active_learning_strategies.fass import FASS
from distil.active_learning_strategies.adversarial_bim import AdversarialBIM
from distil.active_learning_strategies.adversarial_deepfool import AdversarialDeepFool
from distil.active_learning_strategies.core_set import CoreSet
from distil.active_learning_strategies.least_confidence_sampling import LeastConfidenceSampling
from distil.active_learning_strategies.margin_sampling import MarginSampling
from distil.active_learning_strategies.bayesian_active_learning_disagreement_dropout import BALDDropout
from distil.active_learning_strategies.batch_bald import BatchBALDDropout
from distil.utils.train_helper import data_train
from distil.utils.utils import LabeledToUnlabeledDataset

from google.colab import drive
import warnings
warnings.filterwarnings("ignore")


# def write_dict_to_file(my_dict, save_directory, pickle_file_name='my_dict.pkl'):
#     if not os.path.exists(save_directory):
#         os.makedirs(save_directory)
#     file_path = os.path.join(save_directory, pickle_file_name)
#     with open(file_path, 'wb') as f:
#         pickle.dump(my_dict, f)



def write_dict_to_file(my_dict, save_directory, file_name):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    file_path = os.path.join(save_directory, file_name)
    with open(file_path, 'w') as f:
        json.dump(my_dict, f)

def create_indices_dict(initial_train_indices, full_train_dataset, index_map):
    # Initialize dictionary
    my_dict = {
        'indices_selected_original_dataset': {},
        'indices_selected_original_dataset_flat': {},
    }

    # Calculate initial_unlabeled_indices
    initial_unlabeled_indices = list(set(range(len(full_train_dataset))) - set(initial_train_indices))

    # Compute the list of used indices
    _list_indices_used = []
    for i in range(len(index_map) - 1):
        _list_temp = list(set(index_map[i]) - set(index_map[i+1]))
        _list_indices_used.append(_list_temp)

    # Compute selected_indices_original_dataset and add it to the dict
    selected_indices_original_dataset = [[initial_unlabeled_indices[i] for i in indices_list] for indices_list in _list_indices_used]
    selected_indices_original_dataset.insert(0, list(initial_train_indices))

    for i, indices in enumerate(selected_indices_original_dataset):
        round_key = f'Round {i+1}'
        my_dict['indices_selected_original_dataset'][round_key] = indices

    # Compute and add selected_indices_original_dataset_flat to the dict
    selected_indices_original_dataset_flat_list = [item for sublist in selected_indices_original_dataset for item in sublist]
    my_dict['indices_selected_original_dataset_flat'] = selected_indices_original_dataset_flat_list

    
    return my_dict



class Checkpoint:

    def __init__(self, acc_list=None, indices=None, state_dict=None, experiment_name=None, path=None):

        # If a path is supplied, load a checkpoint from there.
        if path is not None:

            if experiment_name is not None:
                self.load_checkpoint(path, experiment_name)
            else:
                raise ValueError("Checkpoint contains None value for experiment_name")

            return

        if acc_list is None:
            raise ValueError("Checkpoint contains None value for acc_list")

        if indices is None:
            raise ValueError("Checkpoint contains None value for indices")

        if state_dict is None:
            raise ValueError("Checkpoint contains None value for state_dict")

        if experiment_name is None:
            raise ValueError("Checkpoint contains None value for experiment_name")

        self.acc_list = acc_list
        self.indices = indices
        self.state_dict = state_dict
        self.experiment_name = experiment_name

    def __eq__(self, other):

        # Check if the accuracy lists are equal
        acc_lists_equal = self.acc_list == other.acc_list

        # Check if the indices are equal
        indices_equal = self.indices == other.indices

        # Check if the experiment names are equal
        experiment_names_equal = self.experiment_name == other.experiment_name

        return acc_lists_equal and indices_equal and experiment_names_equal

    def save_checkpoint(self, path):

        # Get current time to use in file timestamp
        timestamp = time.time_ns()

        # Create the path supplied
        os.makedirs(path, exist_ok=True)

        # Name saved files using timestamp to add recency information
        save_path = os.path.join(path, F"c{timestamp}1")
        copy_save_path = os.path.join(path, F"c{timestamp}2")

        # Write this checkpoint to the first save location
        with open(save_path, 'wb') as save_file:
            pickle.dump(self, save_file)

        # Write this checkpoint to the second save location
        with open(copy_save_path, 'wb') as copy_save_file:
            pickle.dump(self, copy_save_file)

    def load_checkpoint(self, path, experiment_name):

        # Obtain a list of all files present at the path
        timestamp_save_no = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        # If there are no such files, set values to None and return
        if len(timestamp_save_no) == 0:
            self.acc_list = None
            self.indices = None
            self.state_dict = None
            return

        # Sort the list of strings to get the most recent
        timestamp_save_no.sort(reverse=True)

        # Read in two files at a time, checking if they are equal to one another.
        # If they are equal, then it means that the save operation finished correctly.
        # If they are not, then it means that the save operation failed (could not be
        # done atomically). Repeat this action until no possible pair can exist.
        while len(timestamp_save_no) > 1:

            # Pop a most recent checkpoint copy
            first_file = timestamp_save_no.pop(0)

            # Keep popping until two copies with equal timestamps are present
            while True:

                second_file = timestamp_save_no.pop(0)

                # Timestamps match if the removal of the "1" or "2" results in equal numbers
                if (second_file[:-1]) == (first_file[:-1]):
                    break
                else:
                    first_file = second_file

                    # If there are no more checkpoints to examine, set to None and return
                    if len(timestamp_save_no) == 0:
                        self.acc_list = None
                        self.indices = None
                        self.state_dict = None
                        return

            # Form the paths to the files
            load_path = os.path.join(path, first_file)
            copy_load_path = os.path.join(path, second_file)

            # Load the two checkpoints
            with open(load_path, 'rb') as load_file:
                checkpoint = pickle.load(load_file)

            with open(copy_load_path, 'rb') as copy_load_file:
                checkpoint_copy = pickle.load(copy_load_file)

            # Do not check this experiment if it is not the one we need to restore
            if checkpoint.experiment_name != experiment_name:
                continue

            # Check if they are equal
            if checkpoint == checkpoint_copy:

                # This checkpoint will suffice. Populate this checkpoint's fields
                # with the selected checkpoint's fields.
                self.acc_list = checkpoint.acc_list
                self.indices = checkpoint.indices
                self.state_dict = checkpoint.state_dict
                return

        # Instantiate None values in acc_list, indices, and model
        self.acc_list = None
        self.indices = None
        self.state_dict = None

    def get_saved_values(self):

        return (self.acc_list, self.indices, self.state_dict)

def delete_checkpoints(checkpoint_directory, experiment_name):

    # Iteratively go through each checkpoint, deleting those whose experiment name matches.
    timestamp_save_no = [f for f in os.listdir(checkpoint_directory) if os.path.isfile(os.path.join(checkpoint_directory, f))]

    for file in timestamp_save_no:

        delete_file = False

        # Get file location
        file_path = os.path.join(checkpoint_directory, file)

        if not os.path.exists(file_path):
            continue

        # Unpickle the checkpoint and see if its experiment name matches
        with open(file_path, "rb") as load_file:

            checkpoint_copy = pickle.load(load_file)
            if checkpoint_copy.experiment_name == experiment_name:
                delete_file = True

        # Delete this file only if the experiment name matched
        if delete_file:
            os.remove(file_path)

#Logs
def write_logs(logs, save_directory, rd):
  file_path = save_directory + 'run_'+'.txt'
  with open(file_path, 'a') as f:
    f.write('---------------------\n')
    f.write('Round '+str(rd)+'\n')
    f.write('---------------------\n')
    for key, val in logs.items():
      if key == 'Training':
        f.write(str(key)+ '\n')
        for epoch in val:
          f.write(str(epoch)+'\n')
      else:
        f.write(str(key) + ' - '+ str(val) +'\n')


def train_one(full_train_dataset, initial_train_indices, test_dataset, net, n_rounds, budget, args, nclasses, strategy, save_directory, checkpoint_directory, experiment_name, save_dict_directory=None, _list_weights=None):

    # Split the full training dataset into an initial training dataset and an unlabeled dataset
    train_dataset = Subset(full_train_dataset, initial_train_indices)
    initial_unlabeled_indices = list(set(range(len(full_train_dataset))) - set(initial_train_indices))
    unlabeled_dataset = Subset(full_train_dataset, initial_unlabeled_indices)

    # print("unlabeled_dataset (FIRST) INDICES = ", unlabeled_dataset.indices)


    _selected_points_in_full_training_dataset = list(initial_train_indices) #adjust
    _reimaning_points_in_full_training_dataset = unlabeled_dataset.indices


    # Set up the AL strategy
    if strategy == "random":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device']}
        strategy = RandomSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "entropy":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device']}
        strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "margin":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device']}
        strategy = MarginSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "least_confidence":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device']}
        strategy = LeastConfidenceSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "badge":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device']}
        strategy = BADGE(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "coreset":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device']}
        strategy = CoreSet(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "fass":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device']}
        strategy = FASS(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "glister":
        strategy_args = {'batch_size' : args['batch_size'], 'lr': args['lr'], 'device':args['device']}
        strategy = GLISTER(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args, typeOf='rand', lam=0.1)
    elif strategy == "adversarial_bim":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device']}
        strategy = AdversarialBIM(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "adversarial_deepfool":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device']}
        strategy = AdversarialDeepFool(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "bald":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device']}
        strategy = BALDDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
    elif strategy == "batch_bald":
        strategy_args = {'batch_size' : args['batch_size'], 'device':args['device']}
        strategy = BatchBALDDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)        



    logs_dict = {} #len_training_points, test_accuracy, test_selection_time, test_training_time
    aux_training_logs_dict = {} #test_training_logs
    aux_indices_logs_dict = {} #index_map, strategy_select_idx, remaining_unlabeled_indices, used_indices
    aux_torch_datasets_logs_dict = {} # selected_unlabeled_points, train_dataset, unlabeled_dataset


    _temp_logs_index_map = [] #index_map
    _temp_logs_strategy_select_idx = []    #idx
    _temp_logs_selected_unlabeled_points = []  #selected_unlabeled_points
    _temp_logs_train_dataset = [] #train_dataset
    _temp_logs_remaining_unlabeled_indices = []    #remaining_unlabeled_indices
    _temp_logs_unlabeled_dataset = []    #unlabeled_dataset
    _temp_logs_used_indices = [] #used_indices
    _temp_logs_training_points = []
    _temp_logs_test_accuracy = []
    _temp_logs_test_selection_time = []
    _temp_logs_test_training_time = []
    _temp_logs_test_training_logs = []

    _temp_logs_original_dataset_selected_points = []



    # Define acc initially
    acc = np.zeros(n_rounds+1)

    initial_unlabeled_size = len(unlabeled_dataset)

    initial_round = 1

    # Define an index map
    index_map = np.array([x for x in range(initial_unlabeled_size)])

    # Attempt to load a checkpoint. If one exists, then the experiment crashed.
    training_checkpoint = Checkpoint(experiment_name=experiment_name, path=checkpoint_directory)
    rec_acc, rec_indices, rec_state_dict = training_checkpoint.get_saved_values()


    # print("index_map = ", index_map)



    # Check if there are values to recover
    if rec_acc is not None:

        # Restore the accuracy list
        for i in range(len(rec_acc)):
            acc[i] = rec_acc[i]

        # Restore the indices list and shift those unlabeled points to the labeled set.
        index_map = np.delete(index_map, rec_indices)

        # Record initial size of the training dataset
        intial_seed_size = len(train_dataset)

        restored_unlabeled_points = Subset(unlabeled_dataset, rec_indices)
        train_dataset = ConcatDataset([train_dataset, restored_unlabeled_points])

        remaining_unlabeled_indices = list(set(range(len(unlabeled_dataset))) - set(rec_indices))
        unlabeled_dataset = Subset(unlabeled_dataset, remaining_unlabeled_indices)

        # Restore the model
        net.load_state_dict(rec_state_dict)

        # Fix the initial round
        initial_round = (len(train_dataset) - initial_seed_size) // budget + 1

        # Ensure loaded model is moved to GPU
        if torch.cuda.is_available():
            net = net.cuda()

        strategy.update_model(net)
        strategy.update_data(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset))

        dt = data_train(train_dataset, net, args)



    else:

        if torch.cuda.is_available():
            net = net.cuda()

        dt = data_train(train_dataset, net, args)

        acc[0] = dt.get_acc_on_set(test_dataset)
        print('Initial Testing accuracy:', round(acc[0]*100, 2), flush=True)

        logs = {}
        logs['Training Points'] = len(train_dataset)
        logs['Test Accuracy'] =  str(round(acc[0]*100, 2))
        write_logs(logs, save_directory, 0)




        _temp_logs_index_map.append(index_map)
        _temp_logs_strategy_select_idx.append(None)
        _temp_logs_selected_unlabeled_points.append(None)
        _temp_logs_train_dataset.append(train_dataset)
        _temp_logs_remaining_unlabeled_indices.append(None)
        _temp_logs_unlabeled_dataset.append(unlabeled_dataset)
        _temp_logs_used_indices.append(None)
        _temp_logs_training_points.append(len(train_dataset))
        _temp_logs_test_accuracy.append(str(round(acc[0]*100, 2)))
        _temp_logs_test_selection_time.append(None)
        _temp_logs_test_training_time.append(None)
        _temp_logs_test_training_logs.append(None)

        #Updating the trained model in strategy class
        strategy.update_model(net)

    # Record the training transform and test transform for disabling purposes
    train_transform = full_train_dataset.transform
    test_transform = test_dataset.transform

    ##User Controlled Loop
    for rd in range(initial_round, n_rounds+1):
        print('-------------------------------------------------')
        print('Round', rd)
        print('-------------------------------------------------')



        sel_time = time.time()
        full_train_dataset.transform = test_transform # Disable any augmentation while selecting points
        idx = strategy.select(budget)
        full_train_dataset.transform = train_transform # Re-enable any augmentation done during training
        sel_time = time.time() - sel_time
        print("Selection Time:", sel_time)        
        # print("idx === ", idx, "\n")

        selected_unlabeled_points = Subset(unlabeled_dataset, idx)
        # print("selected_unlabeled_points === ", selected_unlabeled_points, "\n")
        train_dataset = ConcatDataset([train_dataset, selected_unlabeled_points])

        remaining_unlabeled_indices = list(set(range(len(unlabeled_dataset))) - set(idx))
        # print("remaining_unlabeled_indices === ", remaining_unlabeled_indices, "\n")
        unlabeled_dataset = Subset(unlabeled_dataset, remaining_unlabeled_indices)
        # print("unlabeled_dataset indices === ", unlabeled_dataset.indices, "\n")

        # Update the index map
        index_map = np.delete(index_map, idx, axis = 0)
        # print("index_map === ", index_map, "\n")

        # print('Number of training points -', len(train_dataset))

        # Start training
        strategy.update_data(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset))
        dt.update_data(train_dataset)
        t1 = time.time()
        clf, train_logs = dt.train(None)
        t2 = time.time()
        acc[rd] = dt.get_acc_on_set(test_dataset)
        logs = {}
        logs['Training Points'] = len(train_dataset)
        logs['Test Accuracy'] =  str(round(acc[rd]*100, 2))
        logs['Selection Time'] = str(sel_time)
        logs['Trainining Time'] = str(t2 - t1)
        logs['Training'] = train_logs
        print("Training Time:", str(t2 - t1))



        def save_model_weights(model):
            """Saves the model weights."""
            return {k: v.clone() for k, v in model.state_dict().items()}

        if _list_weights is not None:
            print("saving weights...")
            _list_weights.append(save_model_weights(clf))



        write_logs(logs, save_directory, rd)
        strategy.update_model(clf)
        print('Testing accuracy:', round(acc[rd]*100, 2), flush=True)

        # Create a checkpoint
        used_indices = np.array([x for x in range(initial_unlabeled_size)])
        used_indices = np.delete(used_indices, index_map).tolist()

        # print("used_indices === ", used_indices, "\n")

        round_checkpoint = Checkpoint(acc.tolist(), used_indices, clf.state_dict(), experiment_name=experiment_name)
        round_checkpoint.save_checkpoint(checkpoint_directory)


        _temp_logs_index_map.append(index_map)
        _temp_logs_strategy_select_idx.append(idx)
        _temp_logs_selected_unlabeled_points.append(selected_unlabeled_points)
        _temp_logs_train_dataset.append(train_dataset)
        _temp_logs_remaining_unlabeled_indices.append(remaining_unlabeled_indices)
        _temp_logs_unlabeled_dataset.append(unlabeled_dataset)
        _temp_logs_used_indices.append(used_indices)
        _temp_logs_training_points.append(len(train_dataset))
        _temp_logs_test_accuracy.append(str(round(acc[rd]*100, 2)))
        _temp_logs_test_selection_time.append(str(sel_time))
        _temp_logs_test_training_time.append(str(t2 - t1) )
        _temp_logs_test_training_logs.append(train_logs)

        _temp_logs_original_dataset_selected_points.append(_selected_points_in_full_training_dataset)



    print('Training Completed')


    logs_dict['len_training_points'] = _temp_logs_training_points
    logs_dict['test_accuracy'] = _temp_logs_test_accuracy
    logs_dict['test_selection_time'] = _temp_logs_test_selection_time
    logs_dict['test_training_time'] = _temp_logs_test_training_time


    aux_indices_logs_dict['index_map'] = _temp_logs_index_map
    aux_indices_logs_dict['strategy_select_idx'] = _temp_logs_strategy_select_idx
    aux_indices_logs_dict['remaining_unlabeled_indices'] = _temp_logs_remaining_unlabeled_indices
    aux_indices_logs_dict['used_indices'] = _temp_logs_used_indices


    aux_training_logs_dict['test_training_logs'] = _temp_logs_test_training_logs


    aux_torch_datasets_logs_dict['selected_unlabeled_points'] = _temp_logs_selected_unlabeled_points
    aux_torch_datasets_logs_dict['train_dataset'] = _temp_logs_train_dataset    
    aux_torch_datasets_logs_dict['unlabeled_dataset'] = _temp_logs_unlabeled_dataset
    
    

    try: 
        write_dict_to_file(logs_dict, save_dict_directory, file_name='logs_dict.json')
    except: 
        print("not possible to write logs_dict")

    try: 
        write_dict_to_file(aux_indices_logs_dict, save_dict_directory, file_name='aux_indices_logs_dict.json')
    except: 
        print("not possible to write aux_indices_logs_dict")

    try: 
        write_dict_to_file(aux_training_logs_dict, save_dict_directory, file_name='aux_training_logs_dict.json')
    except: 
        print("not possible to write aux_training_logs_dict")

    try: 
        write_dict_to_file(aux_torch_datasets_logs_dict, save_dict_directory, file_name='aux_torch_datasets_logs_dict.json')
    except: 
        print("not possible to write aux_torch_datasets_logs_dict")                        


    if save_dict_directory is not None:

        try:

            # ESTA DANDO ERRO AQUI!!!! TEM QUE REVER!!!!

            used_indices_dict = create_indices_dict(
                initial_train_indices = initial_train_indices,
                full_train_dataset = full_train_dataset,
                index_map = logs_dict['index_map'])

            write_dict_to_file(used_indices_dict, save_dict_directory, file_name='used_indices_dict.json')
    
            return acc, logs_dict, used_indices_dict
        except:
            print("Not possible to create used_indices_dict dictionary")
            return acc, _list_weights

    
    return acc, logs_dict, _list_weights        




def get_dataset(data_set_name, dataset_root_path='../downloaded_data/', data_augumentation=False, train_transform=None, test_transform=None, nclasses=None, custom_train_root=None, custom_test_root=None):

    if data_set_name == "CIFAR10":

        if train_transform == None:
            if data_augumentation == True:
                train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            else:
                train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            None

        if test_transform == None:            
                test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            None
   

        full_train_dataset = datasets.CIFAR10(dataset_root_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
        test_dataset = datasets.CIFAR10(dataset_root_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)

        nclasses = 10 # NUM CLASSES HERE

    # elif data_set_name == "CIFAR100":

        # if train_transform == None:
        #     train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        # if test_transform == None:
        #     test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        # full_train_dataset = datasets.CIFAR100(dataset_root_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
        # test_dataset = datasets.CIFAR100(dataset_root_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)

        # nclasses = 100 # NUM CLASSES HERE

    elif data_set_name == "MNIST":

        image_dim=28

        if train_transform == None:
            if data_augumentation == True:
                train_transform = transforms.Compose([transforms.RandomCrop(image_dim, padding=4), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            else:
                train_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
        else:
            None

        if test_transform == None:            
                test_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        else:
            None

        full_train_dataset = datasets.MNIST(dataset_root_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
        test_dataset = datasets.MNIST(dataset_root_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)

        nclasses = 10 # NUM CLASSES HERE

    # elif data_set_name == "FashionMNIST":

        # if train_transform == None:
        #     train_transform = transforms.Compose([transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # if test_transform == None:
        #     test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # Use mean/std of MNIST

        # full_train_dataset = datasets.FashionMNIST(dataset_root_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
        # test_dataset = datasets.FashionMNIST(dataset_root_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)

        # nclasses = 10 # NUM CLASSES HERE

    # elif data_set_name == "SVHN":

        # if train_transform == None:
        #     train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        # if test_transform == None:
        #     test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std

        # full_train_dataset = datasets.SVHN(dataset_root_path, split='train', download=True, transform=train_transform, target_transform=torch.tensor)
        # test_dataset = datasets.SVHN(dataset_root_path, split='test', download=True, transform=test_transform, target_transform=torch.tensor)

        # nclasses = 10 # NUM CLASSES HERE

    # elif data_set_name == "ImageNet":

        # if train_transform == None:
        #     train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        # if test_transform == None:
        #     test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std

        # # Note: Not automatically downloaded due to size restrictions. Notebook needs to be adapted to run on local device.
        # full_train_dataset = datasets.ImageNet(dataset_root_path, download=False, split='train', transform=train_transform, target_transform=torch.tensor)
        # test_dataset = datasets.ImageNet(dataset_root_path, download=False, split='val', transform=test_transform, target_transform=torch.tensor)

        # nclasses = 1000 # NUM CLASSES HERE


    else:
        print("Custom Dataset: ", data_set_name)
        if not (train_transform and test_transform and nclasses and custom_train_root and custom_test_root):
            raise ValueError("For custom datasets, train_transform, test_transform, nclasses, custom_train_root, and custom_test_root must be provided")
        
        full_train_dataset = datasets.ImageFolder(root=custom_train_root, transform=train_transform, target_transform=torch.tensor)
        test_dataset = datasets.ImageFolder(root=custom_test_root, transform=test_transform, target_transform=torch.tensor)

    
    return full_train_dataset, test_dataset, nclasses