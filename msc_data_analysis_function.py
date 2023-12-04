import os
import json
import pickle
import pandas as pd

import torch 
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import cuml

import pandas as pd
import numpy as np



import pickle
import os
import time
import re 

from matplotlib.patches import Rectangle, ConnectionPatch
from importlib import reload

import matplotlib.ticker as ticker


import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle, ConnectionPatch
import matplotlib.colors as mcolors




import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle, ConnectionPatch
import matplotlib.colors as mcolors

import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

from sklearn.metrics import silhouette_score




# extract_run_number
# extract_experiment_name
# get_image_path_and_label
# apply_get_image_path_and_label
# find_paths
# find_subdirectories_names
# process_logs_dict
# process_used_indices_dict
# process_experiment_path
# process_path

# evaluate_tsne_embedding
# optimize_tsne_params
# generate_tsne_embeddings_for_datasets

# extract_number
# extract_run_number
# extract_experiment_name
# rename_dataframe_elements
# create_df_metrics
# timming_table_consolidation
# f_order_kpis
# to_latex_with_multirow_and_lines
# calculate_efficiency
# calculate_efficiencies
# plot_efficiencies
# adjusted_plot_label_efficiency
# adjusted_plot_with_combined_features




#################################  #################################  #################################  #################################
# Process Files (Paths)
#################################  #################################  #################################  #################################

# Function to extract run number
def extract_run_number(experiment_name):
    match = re.search(r'run_(\d+)$', experiment_name)
    return int(match.group(1)) if match else 1

# Function to extract common experiment name
def extract_experiment_name(experiment_name):
    return re.sub(r'_run_\d+$', '', experiment_name)



def get_image_path_and_label(index, dataset):
    # Retrieve the image path and label for the given index
    image_path, label_index = dataset.imgs[index]
    label = dataset.classes[label_index]
    return image_path, label

def apply_get_image_path_and_label(row, datasets_dict):
    dataset_name = row['database']
    if dataset_name not in datasets_dict:
        return pd.Series([None, None, None], index=['image_path', 'image_name', 'label'])
    
    dataset = datasets_dict[dataset_name]
    index = int(row['Selected Indice'])
    image_path, label = get_image_path_and_label(index, dataset)
    image_name = os.path.basename(image_path)
    return pd.Series([image_path, image_name, label], index=['image_path', 'image_name', 'label'])

def find_paths(directory):
    return [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def find_subdirectories_names(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def process_logs_dict(dict_random, database, experiment, strategy):
    rounds = list(range(1, len(dict_random['test_training_time']) + 1))
    dict_list = [
        [database] * len(rounds),
        [experiment] * len(rounds),
        [strategy] * len(rounds),
        rounds,
        dict_random['len_training_points'],
        dict_random['test_accuracy'],
        dict_random['test_selection_time'],
        dict_random['test_training_time'],
    ]

    # Create DataFrame
    df = pd.DataFrame(dict_list)
    df = df.T
    df.columns = ['database', 'experiment', 'strategy', 'rounds', 'len_training_points', 'test_accuracy', 'test_selection_time', 'test_training_time']

    # Convert data types
    df['rounds'] = df['rounds'].astype(int)
    df['len_training_points'] = df['len_training_points'].astype(int)
    df['test_accuracy'] = df['test_accuracy'].astype(float)
    df['test_selection_time'] = df['test_selection_time'].astype(float)
    df['test_training_time'] = df['test_training_time'].astype(float)

    return df

def process_used_indices_dict(dict_indices, database, experiment, strategy):
    round_data = dict_indices['indices_selected_original_dataset']

    rounds = []
    selected_order_continuous = []
    selected_order_per_round = []
    selected_indices = []

    order_counter_continuous = 1

    for round_number, indices in round_data.items():
        # Extract the numeric part from the round_number string
        round_number_int = int(''.join(filter(str.isdigit, round_number)))

        order_counter_per_round = 1
        for index in indices:
            rounds.append(round_number_int)
            selected_order_continuous.append(order_counter_continuous)
            selected_order_per_round.append(order_counter_per_round)
            selected_indices.append(index)

            order_counter_continuous += 1
            order_counter_per_round += 1

    df = pd.DataFrame({
        'database': database,
        'experiment': experiment,
        'strategy': strategy,
        'rounds': rounds,
        'Selected Indice Order (Continuous)': selected_order_continuous,
        'Selected Indice Order (Per Round)': selected_order_per_round,
        'Selected Indice': selected_indices
    })

    df['rounds'] = df['rounds'].astype(int)
    df['Selected Indice Order (Continuous)'] = df['Selected Indice Order (Continuous)'].astype(int)
    df['Selected Indice Order (Per Round)'] = df['Selected Indice Order (Per Round)'].astype(int)
    df['Selected Indice'] = df['Selected Indice'].astype(int)

    return df
    
def process_experiment_path(path):
    split_path = path.split('/')
    _experiment = split_path[-1]
    database_name = split_path[-2]

    _list_paths = find_paths(path)
    _list_strategies = find_subdirectories_names(path)

    _list_dfs_logs = []
    _list_dfs_indices = []
    error_paths = []

    for i in range(len(_list_paths)):
        strategy_path = _list_paths[i]
        strategy = _list_strategies[i]

        # Process logs_dict files
        try:
            pkl_logs_file = os.path.join(strategy_path, 'logs_dict.pkl')
            json_logs_file = os.path.join(strategy_path, 'logs_dict.json')

            if os.path.exists(pkl_logs_file):
                with open(pkl_logs_file, 'rb') as handle:
                    dict_logs = pickle.load(handle)
            elif os.path.exists(json_logs_file):
                with open(json_logs_file, 'r') as handle:
                    dict_logs = json.load(handle)
            else:
                raise FileNotFoundError("No logs_dict file found")

            df_logs = process_logs_dict(dict_logs, database_name, _experiment, strategy)
            _list_dfs_logs.append(df_logs)

        except Exception as e:
            print(f"Error in processing logs_dict: {e}")
            error_paths.append(strategy_path)

        
        # Process used_indices_dict files        
        try:
            pkl_indices_file = os.path.join(strategy_path, 'used_indices_dict.pkl')
            json_indices_file = os.path.join(strategy_path, 'used_indices_dict.json')

            if os.path.exists(pkl_indices_file):
                with open(pkl_indices_file, 'rb') as handle:
                    dict_indices = pickle.load(handle)
            elif os.path.exists(json_indices_file):
                with open(json_indices_file, 'r') as handle:
                    dict_indices = json.load(handle)
            else:
                continue

            df_indices = process_used_indices_dict(dict_indices, database_name, _experiment, strategy)
            _list_dfs_indices.append(df_indices)

        except Exception as e:
            print(f"Error in processing used_indices_dict: {e}")
            error_paths.append(strategy_path)

    if error_paths:
        print("Error paths:", error_paths)

    return _list_dfs_logs, _list_dfs_indices

def process_path(_path, datasets_dict=None):
    dfs_logs = []
    dfs_indices = []

    components = _path.split('/')
    components = [comp for comp in components if comp]

    if 'dict' in components:
        subdirs_beyond_dict = len(components) - components.index('dict') - 1
    else:
        return "Path is not in the expected pattern.", None

    if subdirs_beyond_dict == 0:
        print("This is the dicts path.")
        for database_path in find_paths(_path):
            for experiment_path in find_paths(database_path):
                list_dfs_logs, list_dfs_indices = process_experiment_path(experiment_path)
                dfs_logs.extend(list_dfs_logs)
                dfs_indices.extend(list_dfs_indices)

    elif subdirs_beyond_dict == 1:
        print("This is the database path.")
        for experiment_path in find_paths(_path):
            list_dfs_logs, list_dfs_indices = process_experiment_path(experiment_path)
            dfs_logs.extend(list_dfs_logs)
            dfs_indices.extend(list_dfs_indices)

    elif subdirs_beyond_dict == 2:
        print("This is the experiment path.")
        list_dfs_logs, list_dfs_indices = process_experiment_path(_path)
        dfs_logs.extend(list_dfs_logs)
        dfs_indices.extend(list_dfs_indices)

    else:
        print("Path pattern is not recognized.")
        return None, None

    # Concatenate all DataFrames in each list if they are not empty
    final_df_logs = pd.concat(dfs_logs, ignore_index=True) if dfs_logs else pd.DataFrame()
    final_df_indices = pd.concat(dfs_indices, ignore_index=True) if dfs_indices else pd.DataFrame()

    # Additional processing for df_logs
    if not final_df_logs.empty:
        final_df_logs['labeled_data_added'] = final_df_logs.groupby(['database', 'experiment', 'strategy'])['len_training_points'].diff().fillna(final_df_logs['len_training_points'])

    
    if not final_df_indices.empty:
      if datasets_dict != None:        
        final_df_indices['Selected Indice'] = final_df_indices['Selected Indice'].astype(int)
        # final_df_indices[['image_path', 'image_name', 'label']] = final_df_indices.apply(apply_get_image_path_and_label, axis=1) 
        final_df_indices[['image_path', 'image_name', 'label']] = final_df_indices.apply(lambda row: apply_get_image_path_and_label(row, datasets_dict), axis=1)


    return final_df_logs, final_df_indices



#################################  #################################  #################################  #################################
# Create Pandas DataFrame
#################################  #################################  #################################  #################################


# Function to evaluate t-SNE embeddings using scikit-learn's silhouette_score
def evaluate_tsne_embedding(embedding, labels):
    # Calculate the silhouette score (note: expects NumPy arrays)
    return -silhouette_score(embedding, labels)  # Negative because Optuna minimizes the objective

# Function to optimize t-SNE parameters
def optimize_tsne_params(features_np, labels, trial):
    try:
        n_components = trial.suggest_int("n_components", 2, 2)
        perplexity = trial.suggest_float("perplexity", 5, 50)
        learning_rate = trial.suggest_float("learning_rate", 10, 1000)

        tsne = cuml.TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
        embedding = tsne.fit_transform(features_np)

        # Evaluate the embedding quality
        return evaluate_tsne_embedding(embedding, labels)
    except Exception as e:
        print(f"An error occurred in trial {trial.number}: {e}")
        return float('inf')  # Return a large value to indicate failure

# Function to generate t-SNE embeddings for datasets
def generate_tsne_embeddings_for_datasets(dict_datasets, dict_models, n_trials=10):
    default_model = resnet50(pretrained=True).eval().cuda()
    all_tsne_embeddings = pd.DataFrame()

    for dataset_name, dataset in dict_datasets.items():
        model = dict_models.get(dataset_name, default_model)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Extract features and labels
        features = []
        labels = []
        with torch.no_grad():
            for data in dataloader:
                inputs, batch_labels = data[0].cuda(), data[1]
                outputs = model(inputs)
                features.append(outputs.squeeze(-1).squeeze(-1))
                labels.append(batch_labels)

        features = torch.cat(features, dim=0)
        features_np = features.cpu().numpy()
        labels = torch.cat(labels, dim=0).numpy()

        # Optimize t-SNE parameters
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: optimize_tsne_params(features_np, labels, trial), n_trials=n_trials)

        if len(study.trials) == 0 or study.best_trial is None:
            print(f"No successful trials for dataset {dataset_name}. Using default t-SNE parameters.")
            best_params = {'n_components': 2, 'perplexity': 30, 'learning_rate': 200}
        else:
            best_params = study.best_params


        # best_params =  {'n_components': 2, 'perplexity': 49.25938220738151, 'learning_rate': 642.3745696585185} #345
        # print("best_params = ", best_params)



        tsne = cuml.TSNE(**best_params)
        embedding = tsne.fit_transform(features_np)

        # Create a DataFrame for embeddings
        tsne_df = pd.DataFrame(embedding, columns=['X1', 'X2'])
        tsne_df['Indices'] = range(len(tsne_df))
        tsne_df['database'] = dataset_name

        all_tsne_embeddings = pd.concat([all_tsne_embeddings, tsne_df], ignore_index=True)

    return all_tsne_embeddings




#################################  #################################  #################################  #################################
# Data Analysis
#################################  #################################  #################################  #################################

def extract_number(label):
    match = re.search(r'\d+', label)
    return int(match.group()) if match else 0

def extract_run_number(experiment_name):
    match = re.search(r'run_(\d+)$', experiment_name)
    return int(match.group(1)) if match else 1

def extract_experiment_name(experiment_name):
    return re.sub(r'_run_\d+$', '', experiment_name)

def rename_dataframe_elements(df, dict_renames):
    """
    Rename elements in the DataFrame according to the provided rules.

    :param df: The pandas DataFrame to be renamed.
    :param rename_rules: A dictionary containing renaming rules for strategies, KPIs, and columns.
    :return: A new DataFrame with renamed elements.
    """
    # Create a copy of the DataFrame to avoid modifying the original one
    renamed_df = df.copy()

    if dict_renames:
        # Replace values in the 'strategy' column
        if 'rename_dict_strategies' in dict_renames and 'strategy' in renamed_df.columns:
            try:
                renamed_df['strategy'] = renamed_df['strategy'].replace(dict_renames['rename_dict_strategies'])
            except Exception as e:
                print(f'Error when trying to rename rename_dict_strategies: {e}')

        # Replace values in the 'kpi' column
        if 'rename_dict_kpis' in dict_renames and 'kpi' in renamed_df.columns:
            try:
                renamed_df['kpi'] = renamed_df['kpi'].replace(dict_renames['rename_dict_kpis'])
            except Exception as e:
                print(f'Error when trying to rename rename_dict_kpis: {e}')

        # Rename DataFrame columns
        if 'rename_dict_columns' in dict_renames:
            try:
                renamed_df = renamed_df.rename(columns=dict_renames['rename_dict_columns'])
            except Exception as e:
                print(f'Error when trying to rename rename_dict_columns: {e}')

    return renamed_df

def create_df_metrics(df, cumulative_time=True, dict_renames=None, _additional_dimension_column_name=None):

    if _additional_dimension_column_name is not None:
        _additional_dimension = 'Yes'
    else:
        _additional_dimension = None

    ##################################################################################################################################################################
    # Generate Basic Metrics    
    ##################################################################################################################################################################    
    
    results = []

    # Group by experiment_group, strategy, and rounds
    if _additional_dimension:
        grouped_df = df.groupby(['experiment_group', 'strategy', 'len_training_points',_additional_dimension_column_name])   

        # Loop over each group
        for (experiment_group, strategy, rounds_num, _additional_dimension), group in grouped_df:


            # Calculate individual time metrics
            selection_time = group['test_selection_time'].sum()
            training_time = group['test_training_time'].sum()
            total_time = selection_time + training_time

            total_runs = group['test_selection_time'].shape[0]

            # Calculate cumulative time metrics if required
            
             


            if cumulative_time:
                cumulative_selection_time = df[(df['experiment_group'] == experiment_group) &
                                                (df['strategy'] == strategy) &
                                                (df['len_training_points'] <= rounds_num) & 
                                               (df[_additional_dimension_column_name] == _additional_dimension)
                                              ]['test_selection_time'].sum()
                                              
                cumulative_training_time = df[(df['experiment_group'] == experiment_group) &
                                              (df['strategy'] == strategy) &
                                              (df['len_training_points'] <= rounds_num) & 
                                              (df[_additional_dimension_column_name] == _additional_dimension)
                                              ]['test_training_time'].sum()
                cumulative_total_time = cumulative_selection_time + cumulative_training_time
            else:
                cumulative_selection_time = selection_time
                cumulative_training_time = training_time
                cumulative_total_time = total_time



            # Calculate mean and standard deviation for each time metric
            selection_time_mean = cumulative_selection_time / total_runs
            selection_time_std = group['test_selection_time'].std(ddof=1) if len(group['test_selection_time']) > 1 else 0
            training_time_mean = cumulative_training_time / total_runs
            training_time_std = group['test_training_time'].std(ddof=1) if len(group['test_training_time']) > 1 else 0
            total_time_mean = cumulative_total_time / total_runs
            total_time_std = group['test_training_time'].std(ddof=1) if len(group['test_training_time']) > 1 else 0  # Assuming total time std deviation is similar to training time

            # Calculate mean and standard deviation for accuracy
            accuracy_mean = group['test_accuracy'].mean()
            accuracy_std = group['test_accuracy'].std(ddof=1) if len(group['test_accuracy']) > 1 else 0

            # Append the results
            results.extend([
                [experiment_group, strategy, rounds_num, _additional_dimension, 'selection_time_seconds', selection_time_mean, selection_time_std],
                [experiment_group, strategy, rounds_num, _additional_dimension, 'training_time_seconds', training_time_mean, training_time_std],
                [experiment_group, strategy, rounds_num, _additional_dimension, 'total_time_seconds', total_time_mean, total_time_std],
                [experiment_group, strategy, rounds_num, _additional_dimension, 'accuracy', accuracy_mean, accuracy_std]
            ])

        _columns=['experiment_group', 'strategy', 'len_training_points']
        _columns.extend([_additional_dimension_column_name])
        _columns.extend(['kpi', 'mean_value', 'std_value'])

    

    else:
        grouped_df = df.groupby(['experiment_group', 'strategy', 'len_training_points'])

        # Loop over each group
        for (experiment_group, strategy, rounds_num), group in grouped_df:
            # Calculate individual time metrics
            selection_time = group['test_selection_time'].sum()
            training_time = group['test_training_time'].sum()
            total_time = selection_time + training_time

            total_runs = group['test_selection_time'].shape[0]

            # Calculate cumulative time metrics if required
            if cumulative_time:
                cumulative_selection_time = df[(df['experiment_group'] == experiment_group) &
                                               (df['strategy'] == strategy) &
                                               (df['len_training_points'] <= rounds_num)]['test_selection_time'].sum()
                cumulative_training_time = df[(df['experiment_group'] == experiment_group) &
                                              (df['strategy'] == strategy) &
                                              (df['len_training_points'] <= rounds_num)]['test_training_time'].sum()
                cumulative_total_time = cumulative_selection_time + cumulative_training_time
            else:
                cumulative_selection_time = selection_time
                cumulative_training_time = training_time
                cumulative_total_time = total_time

            # Calculate mean and standard deviation for each time metric
            selection_time_mean = cumulative_selection_time / total_runs
            selection_time_std = group['test_selection_time'].std(ddof=1) if len(group['test_selection_time']) > 1 else 0
            training_time_mean = cumulative_training_time / total_runs
            training_time_std = group['test_training_time'].std(ddof=1) if len(group['test_training_time']) > 1 else 0
            total_time_mean = cumulative_total_time / total_runs
            total_time_std = group['test_training_time'].std(ddof=1) if len(group['test_training_time']) > 1 else 0  # Assuming total time std deviation is similar to training time

            # Calculate mean and standard deviation for accuracy
            accuracy_mean = group['test_accuracy'].mean()
            accuracy_std = group['test_accuracy'].std(ddof=1) if len(group['test_accuracy']) > 1 else 0

            # Append the results
            results.extend([
                [experiment_group, strategy, rounds_num, 'selection_time_seconds', selection_time_mean, selection_time_std],
                [experiment_group, strategy, rounds_num, 'training_time_seconds', training_time_mean, training_time_std],
                [experiment_group, strategy, rounds_num, 'total_time_seconds', total_time_mean, total_time_std],
                [experiment_group, strategy, rounds_num, 'accuracy', accuracy_mean, accuracy_std]
            ])

            _columns = ['experiment_group', 'strategy', 'len_training_points', 'kpi', 'mean_value', 'std_value']

    # Convert results to DataFrame
    df_basic_kpis = pd.DataFrame(results, columns=_columns)    
    

    ##################################################################################################################################################################
    # Calculate Lifts
    ##################################################################################################################################################################    
    
    results_list = []

    def get_final_signal_and_factor(sign_accuracy, sign_time):
        signal_pair = (sign_accuracy, sign_time)
        signal_factor_map = {
            ('+', '+'): ('+', 2),
            ('-', '-'): ('-', 1),
            ('+', '-'): ('+', 5),
            ('-', '+'): ('-', 10),
            # Additions for Neutral cases
            ('Neutral', '+'): ('+', 2),  # Example, adjust as needed
            ('+', 'Neutral'): ('+', 2),  # Example, adjust as needed
            ('Neutral', '-'): ('-', 5),  # Example, adjust as needed
            ('-', 'Neutral'): ('-', 5),  # Example, adjust as needed
            ('Neutral', 'Neutral'): ('+', 0)  # Example, adjust as needed
        }
        return signal_factor_map.get(signal_pair, ('+', 0))  # Default if pair not found


    if _additional_dimension:


        # Iterate over each group of experiment_group and rounds
        for (group, rnd), group_df in df_basic_kpis.groupby(['strategy', 'len_training_points']):
            # Get unique strategies
            strategies = group_df[_additional_dimension_column_name].unique()


            

            # Create all combinations of strategies
            for strat1, strat2 in product(strategies, repeat=2):
                # Filter data for each strategy
                df_strat1 = group_df[group_df[_additional_dimension_column_name] == strat1]
                df_strat2 = group_df[group_df[_additional_dimension_column_name] == strat2]


                # Calculate deltas and round them to 1 decimal place
                delta_total_time = df_strat1[df_strat1['kpi'] == 'total_time_seconds']['mean_value'].values[0] /
                                   df_strat2[df_strat2['kpi'] == 'total_time_seconds']['mean_value'].values[0] - 1

                delta_accuracy = df_strat1[df_strat1['kpi'] == 'accuracy']['mean_value'].values[0] /
                                 df_strat2[df_strat2['kpi'] == 'accuracy']['mean_value'].values[0] - 1

                
                 # Determine the sign for delta_total_time and delta_accuracy
                sign_total_time = "+" if delta_total_time > 0 else ("Neutral" if delta_total_time == 0 else "-")
                sign_accuracy = "+" if delta_accuracy > 0 else ("Neutral" if delta_accuracy == 0 else "-")                


                final_signal, factor = get_final_signal_and_factor(sign_accuracy, sign_total_time)


                # Calculate lift            
                if delta_total_time == 0 and delta_accuracy == 0:
                    lift = 0  
                elif delta_total_time == 0 and delta_accuracy != 0:
                    lift = delta_accuracy  
                elif delta_total_time != 0 and delta_accuracy == 0:
                    lift = delta_total_time  
                else:
                    lift = delta_accuracy / delta_total_time  

                lift *= final_signal
                lift *= factor
                lift = round(lift, 4)


                # Create a new row and add it to the results list
                new_row = {
                    'experiment_group': group_df['experiment_group'].unique()[0],
                    'len_training_points': rnd,
                    'strategy': group,
                    _additional_dimension_column_name: strat1,
                    'compared_strategy': strat2,
                    'sign_delta_total_time': sign_total_time,
                    'sign_delta_accuracy': sign_accuracy,
                    'delta_total_time': delta_total_time,
                    'delta_accuracy': delta_accuracy,
                    'lift': lift
                }
                results_list.append(new_row)

        # Concatenate all results into a DataFrame
        df_lift = pd.concat([pd.DataFrame([row]) for row in results_list], ignore_index=True)
        df_lift_temp = df_lift.copy()


    else:

        # Iterate over each group of experiment_group and rounds
        for (group, rnd), group_df in df_basic_kpis.groupby(['experiment_group', 'len_training_points']):
            # Get unique strategies
            strategies = group_df['strategy'].unique()

            # Create all combinations of strategies
            for strat1, strat2 in product(strategies, repeat=2):
                # Filter data for each strategy
                df_strat1 = group_df[group_df['strategy'] == strat1]
                df_strat2 = group_df[group_df['strategy'] == strat2]

                # Calculate deltas and round them to 1 decimal place
                delta_total_time = df_strat1[df_strat1['kpi'] == 'total_time_seconds']['mean_value'].values[0] /
                                   df_strat2[df_strat2['kpi'] == 'total_time_seconds']['mean_value'].values[0] - 1

                delta_accuracy = df_strat1[df_strat1['kpi'] == 'accuracy']['mean_value'].values[0] /
                                 df_strat2[df_strat2['kpi'] == 'accuracy']['mean_value'].values[0] - 1

                
                 # Determine the sign for delta_total_time and delta_accuracy
                sign_total_time = "+" if delta_total_time > 0 else ("Neutral" if delta_total_time == 0 else "-")
                sign_accuracy = "+" if delta_accuracy > 0 else ("Neutral" if delta_accuracy == 0 else "-")                


                final_signal, factor = get_final_signal_and_factor(sign_accuracy, sign_total_time)

                # Calculate lift            
                if delta_total_time == 0 and delta_accuracy == 0:
                    lift = 0  
                elif delta_total_time == 0 and delta_accuracy != 0:
                    lift = delta_accuracy  
                elif delta_total_time != 0 and delta_accuracy == 0:
                    lift = delta_total_time  
                else:
                    lift = delta_accuracy / delta_total_time  

                lift *= final_signal
                lift *= factor
                lift = round(lift, 4)


                # Create a new row and add it to the results list
                new_row = {
                    'experiment_group': group,
                    'len_training_points': rnd,
                    'strategy': strat1,
                    'compared_strategy': strat2,
                    'sign_delta_total_time': sign_total_time,
                    'sign_delta_accuracy': sign_accuracy,
                    'delta_total_time': delta_total_time,
                    'delta_accuracy': delta_accuracy,
                    'lift': lift
                }
                results_list.append(new_row)

        # Concatenate all results into a DataFrame
        df_lift = pd.concat([pd.DataFrame([row]) for row in results_list], ignore_index=True)
        df_lift_temp = df_lift.copy()



    ##################################################################################################################################################################
    # Combine df's
    ##################################################################################################################################################################            

    if _additional_dimension:

        df_basic_kpis['compared_strategy'] = '-'
        df_basic_kpis = df_basic_kpis[['experiment_group', 'strategy', 'compared_strategy', 'len_training_points', _additional_dimension_column_name, 'kpi', 'mean_value', 'std_value']]

        df_lift['kpi']  = 'lift'
        df_lift['mean_value']  = df_lift['lift']
        df_lift['std_value']  = None
        df_lift = df_lift[['experiment_group', 'strategy', 'compared_strategy', 'len_training_points', _additional_dimension_column_name, 'kpi', 'mean_value', 'std_value']]
        
        df_lift['compared_strategy']        

        if _additional_dimension_column_name == 'epochs':            
            # def extract_number(label):
            #     match = re.search(r'\d+', label)
            #     return int(match.group()) if match else 0

            # df_lift = df_lift[df_lift['compared_strategy'] == df_lift['compared_strategy'].apply(extract_number).max()]
            df_lift = df_lift[df_lift['compared_strategy'] == df_lift['compared_strategy'].max()]
        else:
            df_lift = df_lift[df_lift['compared_strategy'] == df_lift['compared_strategy'].min()]


    else:
        df_basic_kpis['compared_strategy'] = '-'
        df_basic_kpis = df_basic_kpis[['experiment_group', 'strategy', 'compared_strategy', 'len_training_points', 'kpi', 'mean_value', 'std_value']]

        df_lift['kpi']  = 'lift'
        df_lift['mean_value']  = df_lift['lift']
        df_lift['std_value']  = None
        df_lift = df_lift[['experiment_group', 'strategy', 'compared_strategy', 'len_training_points', 'kpi', 'mean_value', 'std_value']]
        df_lift = df_lift[df_lift['compared_strategy'] == 'random']

    merged_df = pd.concat([df_basic_kpis, df_lift])
    merged_df = merged_df.reset_index(drop=True)


    ##################################################################################################################################################################
    # Final Data Frame
    ##################################################################################################################################################################            

    return merged_df, df_lift_temp, df_basic_kpis

def timming_table_consolidation(df, opt_visualize_std=False, _additional_dimension_column_name=None):


    if _additional_dimension_column_name:


        if opt_visualize_std:
            # Create pivot tables for mean and standard deviation values
            mean_pivot = df.pivot_table(index=['experiment_group', 'strategy', 'kpi', _additional_dimension_column_name], columns='len_training_points', values='mean_value')
            std_pivot = df.pivot_table(index=['experiment_group', 'strategy', 'kpi', _additional_dimension_column_name], columns='len_training_points', values='std_value')

            # Round the mean values
            mean_pivot = mean_pivot.round(1)

            # Function to combine mean and standard deviation for 'accuracy'
            def combine_mean_std(row):
                return pd.Series({mean_col: "{} ± {}".format(row[mean_col], round(std_pivot.loc[row.name, mean_col], 4))
                                  if row.name[2] == 'accuracy' and pd.notna(std_pivot.loc[row.name, mean_col])
                                  else row[mean_col]
                                  for mean_col in mean_pivot.columns})

            # Apply the combine_mean_std function
            combined_pivot = mean_pivot.apply(combine_mean_std, axis=1)
            combined_pivot.columns = [f'{col} points' for col in combined_pivot.columns]
            combined_pivot = combined_pivot.reset_index()


            return combined_pivot

        else:
            # Process for when opt_visualize_std is False
            kpi_order = ['selection_time_seconds', 'training_time_seconds', 'total_time_seconds', 'accuracy', 'lift']

            df_melted = df.melt(id_vars=['experiment_group', 'strategy', 'len_training_points', 'kpi', _additional_dimension_column_name], value_vars=['mean_value'])
            df_melted['value'] = df_melted['value'].round(1)
            df_melted['kpi'] = pd.Categorical(df_melted['kpi'], categories=kpi_order, ordered=True)

            pivot_table = df_melted.pivot_table(index=['experiment_group', 'strategy', 'kpi', _additional_dimension_column_name], columns='len_training_points', values='value')
            pivot_table.columns = [f'{col} points' for col in pivot_table.columns]
            pivot_table = pivot_table.reset_index()
            pivot_table = pivot_table.sort_values(by=[_additional_dimension_column_name, 'strategy', 'experiment_group'], ascending=[True, False, False])

            return pivot_table



    else:

        if opt_visualize_std:
            # Create pivot tables for mean and standard deviation values
            mean_pivot = df.pivot_table(index=['experiment_group', 'strategy', 'kpi'], columns='len_training_points', values='mean_value')
            std_pivot = df.pivot_table(index=['experiment_group', 'strategy', 'kpi'], columns='len_training_points', values='std_value')

            # Round the mean values
            mean_pivot = mean_pivot.round(1)

            # Function to combine mean and standard deviation for 'accuracy'
            def combine_mean_std(row):
                return pd.Series({mean_col: "{} ± {}".format(row[mean_col], round(std_pivot.loc[row.name, mean_col], 4))
                                  if row.name[2] == 'accuracy' and pd.notna(std_pivot.loc[row.name, mean_col])
                                  else row[mean_col]
                                  for mean_col in mean_pivot.columns})

            # Apply the combine_mean_std function
            combined_pivot = mean_pivot.apply(combine_mean_std, axis=1)
            combined_pivot.columns = [f'{col} points' for col in combined_pivot.columns]
            combined_pivot = combined_pivot.reset_index()


            return combined_pivot

        else:
            # Process for when opt_visualize_std is False
            kpi_order = ['selection_time_seconds', 'training_time_seconds', 'total_time_seconds', 'accuracy', 'lift']

            df_melted = df.melt(id_vars=['experiment_group', 'strategy', 'len_training_points', 'kpi'], value_vars=['mean_value'])
            df_melted['value'] = df_melted['value'].round(1)
            df_melted['kpi'] = pd.Categorical(df_melted['kpi'], categories=kpi_order, ordered=True)

            pivot_table = df_melted.pivot_table(index=['experiment_group', 'strategy', 'kpi'], columns='len_training_points', values='value')
            pivot_table.columns = [f'{col} points' for col in pivot_table.columns]
            pivot_table = pivot_table.reset_index()
            pivot_table = pivot_table.sort_values(by=['strategy', 'experiment_group'], ascending=[False, False])

            return pivot_table

def f_order_kpis(df, _kpi_column, _kpi_order, _sort_values_by, _ascending_values):  

    def get_kpi_order(kpi):
        try:
            return _kpi_order.index(kpi)
        except ValueError:
            return len(_kpi_order)  # Assign a large index to unknown or new KPIs

    
    _sort_values_by = ['kpi_sort_index'] + _sort_values_by
    _ascending_values = [True] + _ascending_values

    # Sort the DataFrame using the custom function
    df = df.assign(
        kpi_sort_index=df[_kpi_column].apply(get_kpi_order)
    ).sort_values(
        by=_sort_values_by,
        ascending=_ascending_values
    ).drop(columns='kpi_sort_index')

    return df

def to_latex_with_multirow_and_lines(df, merge_column):
    unique_values = df[merge_column].unique()
    # Start the tabular and make column names bold
    latex_str = "\\begin{tabular}{" + "l" * len(df.columns) + "}\n\\toprule\n"
    latex_str += " & ".join(["\\textbf{" + col + "}" for col in df.columns]) + " \\\\\n\\midrule\n"

    for value in unique_values:
        subset = df[df[merge_column] == value]
        row_span = len(subset)

        first_row = True
        for _, row in subset.iterrows():
            if first_row:
                latex_str += "\\multirow{" + str(row_span) + "}{*}{" + value.replace("%", "\\%") + "} & "
                first_row = False
            else:
                latex_str += " & "

            latex_str += " & ".join([str(row[col]) for col in df.columns if col != merge_column]) + " \\\\\n"

        # Add a horizontal line after each metric group
        latex_str += "\\midrule\n"

    latex_str += "\\bottomrule\n\\end{tabular}"
    return latex_str


def calculate_efficiency(df_active, df_random):
    efficiency_list = []
    accuracy_list = sorted(set(df_active['test_accuracy'].tolist() + df_random['test_accuracy'].tolist()))
    for accuracy in accuracy_list:
        al_rows = df_active[df_active['test_accuracy'] >= accuracy]
        if al_rows.empty:
            efficiency_list.append(None)
            continue
        al_index = int(al_rows.iloc[0]['len_training_points'])

        rs_rows = df_random[df_random['test_accuracy'] >= accuracy]
        if rs_rows.empty:
            efficiency_list.append(None)
            continue
        rs_index = int(rs_rows.iloc[0]['len_training_points'])

        efficiency = rs_index / al_index if al_index != 0 else np.inf
        efficiency_list.append(efficiency)
    return accuracy_list, efficiency_list

def calculate_efficiencies(df_active, df_random):

    strategies = df_active['strategy'].unique()
    efficiency_dict = {}
    accuracy_dict = {}

    for strategy in strategies:
        print("running for .. ", strategy)
        df_active_strategy = df_active[df_active['strategy'] == strategy]
        accuracy_list, efficiency_list = calculate_efficiency(df_active_strategy, df_random)
        efficiency_dict[strategy] = efficiency_list
        accuracy_dict[strategy] = accuracy_list

    return accuracy_dict, efficiency_dict

def plot_efficiencies(accuracy_dict, efficiency_dict, colors_dict=None, opt_fontsize=None):

    plt.figure(figsize=(10,6))

    for strategy in accuracy_dict.keys():
        if colors_dict and strategy in colors_dict:
            color = colors_dict[strategy]
        else:
            color = None  # Default color if not specified

        # Filter out Nones
        valid_indices = [i for i, x in enumerate(efficiency_dict[strategy]) if x is not None]
        valid_accuracy = [accuracy_dict[strategy][i] for i in valid_indices]
        valid_efficiency = [efficiency_dict[strategy][i] for i in valid_indices]

        plt.plot(valid_accuracy, valid_efficiency, marker='o', label=strategy, color=color)

    plt.xlabel('Accuracy', fontsize=opt_fontsize)
    plt.ylabel('Labelling Efficiency', fontsize=opt_fontsize)
    plt.title('Labelling Efficiency vs Accuracy',fontsize=opt_fontsize+2)
    plt.legend()
    plt.grid(True)
    plt.show()

def adjusted_plot_label_efficiency(df, ax, colors_dict=None, std_dev=None, opt_fontsize=12, opt_show_legend=False):



    # Set the background color of the axes
    # ax.set_facecolor('white')

    # Change the color and alpha of the grid
    ax.grid(True, color='gray', alpha=0.5)

    df_random = df[df['strategy'] == 'random']
    df_active = df[df['strategy'] != 'random']

    accuracy_dict, efficiency_dict = calculate_efficiencies(df_active, df_random)


    ##############################################################################################################
    # Initialize the minimum test accuracy for deviation to infinity
    min_test_accuracy_for_deviation = float('inf')

    # Loop through each strategy and its efficiencies
    for strategy, efficiencies in efficiency_dict.items():
        for i, efficiency in enumerate(efficiencies):
            if efficiency is not None and efficiency != 1.0:
                # Find the corresponding test accuracy value
                test_accuracy_value = accuracy_dict[strategy][i]
                # Update the minimum test accuracy if this is the earliest deviation from 1.0
                if test_accuracy_value < min_test_accuracy_for_deviation:
                    min_test_accuracy_for_deviation = test_accuracy_value
                break  # Stop checking once the first deviation is found for this strategy

    # If no deviation is found, use the overall minimum test accuracy
    if min_test_accuracy_for_deviation == float('inf'):
        min_test_accuracy_for_deviation = df['test_accuracy'].min()
    else:
        # Subtract a buffer from the minimum value (e.g., 1 or any value that makes sense for your data)
        buffer = 1
        min_test_accuracy_for_deviation -= buffer

    # Find the maximum test accuracy from the DataFrame
    max_test_accuracy = df['test_accuracy'].max()

    ##############################################################################################################



    for strategy in accuracy_dict.keys():
        if colors_dict and strategy in colors_dict:
            color = colors_dict[strategy]
        else:
            color = None  # Default color if not specified

        valid_indices = [i for i, x in enumerate(efficiency_dict[strategy]) if x is not None]
        valid_accuracy = [accuracy_dict[strategy][i] for i in valid_indices]
        valid_efficiency = [efficiency_dict[strategy][i] for i in valid_indices]

        ax.plot(valid_accuracy, valid_efficiency, lw=1.5, marker="o", label=strategy, color=color, linestyle='-')


        if std_dev and std_dev in df.columns:
            subset = df[df['strategy'] == strategy]
            ax.fill_between(subset['test_accuracy'], subset['efficiency'] - subset[std_dev], subset['efficiency'] + subset[std_dev], color=color, alpha=0.2, marker="o", markersize=4)

    # Manually add the 'Random' strategy
    if 'random' in colors_dict:
        random_color = colors_dict['random']
    else:
        random_color = 'black'

    all_accuracies = sorted(set(sum(accuracy_dict.values(), [])))
    ax.plot(all_accuracies, [1.0] * len(all_accuracies), label='random', color=random_color, lw=1.5, markersize=4) #marker="o"

    ax.set_xlabel('Test Accuracy', fontsize=opt_fontsize)
    ax.set_ylabel('Labelling Efficiency', fontsize=opt_fontsize)
    ax.set_title('Labelling Efficiency vs. Accuracy', fontsize=opt_fontsize+2)



    # Adjust x-axis to start from the first data point and set custom ticks
    # min_test_accuracy = df['test_accuracy'].min()
    # max_test_accuracy = df['test_accuracy'].max()
    # ax.set_xlim(left=0, right=max_len_training_points)
    # ax.set_xlim(left=min_test_accuracy, right=max_test_accuracy)
    ax.set_xlim(left=min_test_accuracy_for_deviation, right=max_test_accuracy)





    # Define the order of strategies as they should appear in the legend
    legend_order = sorted(df['strategy'].unique(), reverse=True)

    # Get handles and labels from the current plot
    handles, labels = ax.get_legend_handles_labels()

    # Sort handles and labels according to the defined order
    handles_labels_sorted = sorted(zip(handles, labels), key=lambda x: legend_order.index(x[1]) if x[1] in legend_order else -1)
    sorted_handles, sorted_labels = zip(*handles_labels_sorted)

    # Create the legend with the sorted handles and labels
    if opt_show_legend:
      ax.legend(sorted_handles, sorted_labels, fontsize=opt_fontsize-2)
    ax.tick_params(axis='both', which='major', labelsize=opt_fontsize-2)



    ax.grid(True)            

def adjusted_plot_with_combined_features(df, ax, zoom_regions=None, std_dev=None, colors_dict=None, opt_fontsize=12, opt_show_legend=False, opt_title=None, use_distinct_colors=False, use_color_variations=False, legend_column_name=None):

    def thousands_formatter(x, pos):
        """Converts large numbers to 'k' notation for thousands, and leaves smaller numbers unchanged."""
        if x >= 1000:
            return '{:.1f}k'.format(float(x / 1000))
        else:
            return '{:.0f}'.format(int(x))            


    def adjust_color(color, factor=0.2):
        """Adjusts the brightness of a color."""
        return tuple(min(max(comp + factor * (0.5 - comp), 0), 1) for comp in color)

    def mix_colors(color1, color2, alpha=0.5):
        """Mixes two colors."""
        color1_rgba = mcolors.to_rgba(color1)
        color2_rgba = mcolors.to_rgba(color2)
        return mcolors.to_hex([(1 - alpha) * c1 + alpha * c2 for c1, c2 in zip(color1_rgba[:3], color2_rgba[:3])])

    def create_color_variations(colors_dict, num_shades=3):
        """Creates variations of the given colors."""
        base_colors = ['white', 'black', 'gray']  # Base colors for mixing
        color_variations = {}
        for strategy, color in colors_dict.items():
            if color in ['black', '#000000', (0, 0, 0)]:
                color_variations[strategy] = ['black'] * num_shades
                continue
            shades = [mix_colors(color, base_color, alpha=i / (num_shades - 1)) for base_color in base_colors for i in range(num_shades)]
            color_variations[strategy] = [color] + shades
        return color_variations

    def get_line_style(experiment_group):
        """Defines line styles based on the experiment group."""
        # styles = ['-', '--', '-.', ':']
        styles = ['-']
        return styles[hash(experiment_group) % len(styles)]

    ax.grid(True, color='gray', alpha=0.5)

    if colors_dict is None:
        colors_dict = dict(zip(df['strategy'].unique(), sns.color_palette(n_colors=len(df['strategy'].unique()))))

    if use_distinct_colors:
        distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i, strategy in enumerate(colors_dict.keys()):
            colors_dict[strategy] = distinct_colors[i % len(distinct_colors)]

    if use_color_variations:
        colors_dict = create_color_variations(colors_dict)

    colors_dict['random'] = 'black'

    if legend_column_name:

      for (strategy, experiment_group), group_data in df.groupby([legend_column_name, 'experiment_group']):
          color = colors_dict[strategy]
          line_style = get_line_style(experiment_group) if legend_column_name in df.columns else '-'
          sns.lineplot(data=group_data, x="len_training_points", y="test_accuracy", lw=2, marker="o", label=strategy, color=color, linestyle=line_style, ax=ax)    

    else:

      for (strategy, experiment_group), group_data in df.groupby(['strategy', 'experiment_group']):
          color = colors_dict[strategy]
          line_style = get_line_style(experiment_group) if 'strategy' in df.columns else '-'
          sns.lineplot(data=group_data, x="len_training_points", y="test_accuracy", lw=2, marker="o", label=strategy, color=color, linestyle=line_style, ax=ax)



    ax.set_xlim(left=df['len_training_points'].min(), right=df['len_training_points'].max())
    ax.xaxis.set_major_locator(ticker.MultipleLocator(df['len_training_points'].min()))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))

    ax.set_xlabel('Labeled Set Size', fontsize=opt_fontsize)
    ax.set_ylabel('Test Accuracy', fontsize=opt_fontsize)
    ax.set_title(opt_title if opt_title else 'AL Strategies Comparison', fontsize=opt_fontsize + 2)

    if opt_show_legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=opt_fontsize - 4, frameon=True)
    else:
        ax.legend_.remove() if ax.legend_ else None

    
    if legend_column_name:    
      legend_order = sorted(df[legend_column_name].unique(), reverse=False)

    else:    
      legend_order = sorted(df['strategy'].unique(), reverse=True)

    # Get handles and labels from the current plot
    handles, labels = ax.get_legend_handles_labels()

    # Sort handles and labels according to the defined order
    handles_labels_sorted = sorted(zip(handles, labels), key=lambda x: legend_order.index(x[1]) if x[1] in legend_order else -1)
    sorted_handles, sorted_labels = zip(*handles_labels_sorted)

    # Create the legend with the sorted handles and labels
    if opt_show_legend:
        ax.legend(sorted_handles, sorted_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=opt_fontsize - 4, frameon=True)
    else:
        ax.legend_.remove() if ax.legend_ else None

    ax.tick_params(axis='both', which='major', labelsize=opt_fontsize - 2)
    ax.grid(True)



    if zoom_regions:
        for region in zoom_regions:
            x1, x2, y1, y2 = region['x1'], region['x2'], region['y1'], region['y2']
            pos = region['pos']

            ax_zoom = ax.inset_axes(pos)

            if legend_column_name:
                sns.lineplot(data=df, x="len_training_points", y="test_accuracy", hue=legend_column_name, palette=colors_dict, lw=2, marker="o", ax=ax_zoom, legend=False)
            else:
                sns.lineplot(data=df, x="len_training_points", y="test_accuracy", hue="strategy", palette=colors_dict, lw=2, marker="o", ax=ax_zoom, legend=False)


            if std_dev and std_dev in df.columns:
                for strategy, color in colors_dict.items():
                    subset = df[df['strategy'] == strategy]
                    ax_zoom.fill_between(subset['len_training_points'], subset['test_accuracy'] - subset[std_dev], subset['test_accuracy'] + subset[std_dev], color=color, alpha=0.2)

            ax_zoom.set_xlim(x1, x2)
            ax_zoom.set_ylim(y1, y2)
            # ax_zoom.set_xticks([])
            # ax_zoom.set_yticks([])
            ax_zoom.set_xlabel('')
            ax_zoom.set_ylabel('')
            # ax_zoom.set_facecolor('whitesmoke')

            ax_zoom.grid(True, color='gray', alpha=0.3)

            # Draw rectangle in the main plot
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='lightgray', linestyle='dashed')
            ax.add_patch(rect)

            # Add connection lines
            con1 = ConnectionPatch(xyA=(x1, y1), xyB=(pos[0], pos[1]), coordsA="data", coordsB="axes fraction", axesA=ax, axesB=ax, color='lightgray')
            con2 = ConnectionPatch(xyA=(x2, y1), xyB=(pos[0] + pos[2], pos[1]), coordsA="data", coordsB="axes fraction", axesA=ax, axesB=ax, color='lightgray')

            ax.add_artist(con1)
            ax.add_artist(con2)
