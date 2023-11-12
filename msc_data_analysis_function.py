import os
import json
import pickle
import pandas as pd



def get_image_path_and_label(index, dataset):
    # Retrieve the image path and label for the given index
    image_path, label_index = dataset.imgs[index]
    label = dataset.classes[label_index]
    return image_path, label

def apply_get_image_path_and_label(row):
    # Convert the index to integer if it's not already
    index = int(row['Selected Indice'])
    image_path, label = get_image_path_and_label(index, dataset)
    # Extract image name from the image path
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

def process_path(_path):
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

    # Additional processing for df_indices
    if not final_df_indices.empty:
        final_df_indices['Selected Indice'] = final_df_indices['Selected Indice'].astype(int)
        final_df_indices[['image_path', 'image_name', 'label']] = final_df_indices.apply(apply_get_image_path_and_label, axis=1)

    return final_df_logs, final_df_indices
