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

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import os
import time

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, ConnectionPatch
from importlib import reload

import matplotlib.ticker as ticker



#################################  #################################  #################################  #################################
# Process Files (Paths)
#################################  #################################  #################################  #################################

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

def generate_tsne_embeddings_for_datasets(dict_datasets, train_transform=None, test_transform=None, model=None, use_cuml=False):
    if model is None:
        model = resnet50(pretrained=True).eval().cuda()
        # if torch.cuda.is_available():
        #     model = model.cuda()

    # Assuming the model is a feature extractor
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])



    features = []

    all_tsne_embeddings = pd.DataFrame()

    for dataset_name, dataset in dict_datasets.items():
        # Data loader for the dataset
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Extract features
        features = []
        with torch.no_grad():
            for data in dataloader:
                inputs = data[0].cuda()
                # if torch.cuda.is_available():
                #     inputs = inputs.cuda()
                outputs = feature_extractor(inputs)
                features.append(outputs.squeeze(-1).squeeze(-1))

        features = torch.cat(features, dim=0)
        features_np = features.cpu().numpy()


        tsne = cuml.TSNE(n_components=2, perplexity=30, learning_rate=200)
        embedding = tsne.fit_transform(features_np)

        # Create a DataFrame for embeddings
        tsne_df = pd.DataFrame(embedding, columns=['X1', 'X2'])
        tsne_df['Indices'] = range(len(tsne_df))
        tsne_df['database'] = dataset_name  # Add the dataset name

        all_tsne_embeddings = pd.concat([all_tsne_embeddings, tsne_df], ignore_index=True)

    return all_tsne_embeddings



#################################  #################################  #################################  #################################
# Data Analysis
#################################  #################################  #################################  #################################

# Accuracy Chart
def adjusted_plot_with_arrow_to_zoom(df, ax, zoom_regions=None, std_dev=None, colors_dict=None):
    
    # sns.set_style("whitegrid", {'axes.grid': True, 'grid.color': '.2'})
    #  ax.grid(True, color='lightgray')


    ax.grid(True, color='gray', alpha=0.5)      

    if colors_dict is None:
        colors_dict = dict(zip(df['strategy'].unique(), sns.color_palette(n_colors=len(df['strategy'].unique()))))

    colors_dict['random'] = 'black'

    # random_data = df[df['strategy'] == 'random']
    # ax.plot(random_data['len_training_points'], random_data['test_accuracy'], color='black', lw=3, marker='o', label='random')
    sns.lineplot(data=df, x="len_training_points", y="test_accuracy", hue="strategy", palette=colors_dict, lw=1, marker="o", ax=ax)

    # ax.legend(title='Strategy')

    # if args is None:
    #   args = args['nothing'] = None

    # if args['chart_title']:
    #   ax.set_title(args['chart_title'])

    # if args['set_x_label']:
    #   ax.set_xlabel(args['set_x_label'])

    # if args['set_ylabel']:
    #   ax.set_xlabel(args['set_ylabel'])

    # if args['legend']:
    #   ax.legend(args['legend'])

    # # ax.tick_params(labelsize=13)
    # ax.grid(True)



    # Adjust x-axis to start from the first data point and set custom ticks
    min_len_training_points = df['len_training_points'].min()
    max_len_training_points = df['len_training_points'].max()
    ax.set_xlim(left=0, right=max_len_training_points)

    # Set custom ticks (e.g., starting from 2000 and incrementing appropriately)
    # Modify this part based on how you want to set your ticks
    tick_interval = 2000  # Set this based on your data
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))


    ax.set_xlabel('Labeled Set Size')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('AL Strategies Comparison')
    ax.legend()
    ax.grid(True)

    if std_dev and std_dev in df.columns:
        for strategy, color in colors_dict.items():
            subset = df[df['strategy'] == strategy]
            ax.fill_between(subset['len_training_points'], subset['test_accuracy'] - subset[std_dev], subset['test_accuracy'] + subset[std_dev], color=color, alpha=0.2)


    if zoom_regions:
        for region in zoom_regions:
            x1, x2, y1, y2 = region['x1'], region['x2'], region['y1'], region['y2']
            pos = region['pos']

            ax_zoom = ax.inset_axes(pos)

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


# Label Efficiency
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

def plot_efficiencies(accuracy_dict, efficiency_dict, colors_dict=None):

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

    plt.xlabel('Accuracy')
    plt.ylabel('Labelling Efficiency')
    plt.title('Labelling Efficiency vs Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_label_efficiency(df, colors_dict=None):

  df_random = _temp_df[_temp_df['strategy'] == 'random']
  df_active = _temp_df[_temp_df['strategy'] != 'random']

  # calculate_efficiencies
  accuracy_dict, efficiency_dict = calculate_efficiencies(df_active, df_random)
  plot_efficiencies(accuracy_dict, efficiency_dict, colors_dict)

def adjusted_plot_label_efficiency(df, ax, colors_dict=None, std_dev=None):

    # Set the background color of the axes
    # ax.set_facecolor('white')      

    # Change the color and alpha of the grid    
    ax.grid(True, color='gray', alpha=0.5)    

    df_random = df[df['strategy'] == 'random']
    df_active = df[df['strategy'] != 'random']

    accuracy_dict, efficiency_dict = calculate_efficiencies(df_active, df_random)

    for strategy in accuracy_dict.keys():
        if colors_dict and strategy in colors_dict:
            color = colors_dict[strategy]
        else:
            color = None  # Default color if not specified

        valid_indices = [i for i, x in enumerate(efficiency_dict[strategy]) if x is not None]
        valid_accuracy = [accuracy_dict[strategy][i] for i in valid_indices]
        valid_efficiency = [efficiency_dict[strategy][i] for i in valid_indices]

        ax.plot(valid_accuracy, valid_efficiency, lw=1, marker="o", label=strategy, color=color)

        if std_dev and std_dev in df.columns:
            subset = df[df['strategy'] == strategy]
            ax.fill_between(subset['test_accuracy'], subset['efficiency'] - subset[std_dev], subset['efficiency'] + subset[std_dev], color=color, alpha=0.2, marker="o", markersize=4)

    # Manually add the 'Random' strategy
    if 'random' in colors_dict:
        random_color = colors_dict['random']
    else:
        random_color = 'black'

    all_accuracies = sorted(set(sum(accuracy_dict.values(), [])))
    ax.plot(all_accuracies, [1.0] * len(all_accuracies), label='random', color=random_color, lw=1, marker="o", markersize=4)

    ax.set_xlabel('Test Accuracy')
    ax.set_ylabel('Labelling Efficiency')
    ax.set_title('Labelling Efficiency vs. Accuracy')




    # Define the order of strategies as they should appear in the legend
    legend_order = sorted(df['strategy'].unique(), reverse=True)

    # Get handles and labels from the current plot
    handles, labels = ax.get_legend_handles_labels()

    # Sort handles and labels according to the defined order
    handles_labels_sorted = sorted(zip(handles, labels), key=lambda x: legend_order.index(x[1]) if x[1] in legend_order else -1)
    sorted_handles, sorted_labels = zip(*handles_labels_sorted)

    # Create the legend with the sorted handles and labels
    ax.legend(sorted_handles, sorted_labels)

    ax.grid(True)            


# Time Calculation
def df_grouped_by_time(df):


    def convert_to_hours_and_minutes(value):
        if isinstance(value, str):
            return value

        hours = value // 60
        minutes = value % 60

        if hours > 0:
            return f"{int(hours)} Hour(s) and {int(minutes)} Minute(s)"
        else:
            return f"{int(minutes)} Minute(s)"


    def p80(x):
        return np.percentile(x, 80)

    grouped = df.groupby(['experiment', 'strategy']).agg({
        'test_selection_time': ['sum'],
        'test_training_time': ['sum'],
        'test_accuracy': ['max', 'mean', 'median']
    })

    # Convert seconds to minutes
    grouped['test_selection_time'] = grouped['test_selection_time'] / 60
    grouped['test_training_time'] = grouped['test_training_time'] / 60

    # Reformat the column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped.reset_index(inplace=True)

    # Rename the columns
    grouped.rename(columns={
        'test_selection_time_sum': 'Total Selection Time (minutes)',
        'test_training_time_sum': 'Total Training Time (minutes)',
        'test_accuracy_max': 'Max Accuracy',
        'test_accuracy_mean': 'Average Accuracy',
        'test_accuracy_median': 'Median Accuracy'
    }, inplace=True)

    # Rounding and converting to integer
    for column in grouped.columns[2:]:
        if 'Time' in column:
            grouped[column] = grouped[column].round().astype('Int64')
            grouped[column] = grouped[column].apply(convert_to_hours_and_minutes)
        else:
            grouped[column] = grouped[column].round(2)

    # Replace NaNs with 'Not Available'
    grouped.fillna('Not Available', inplace=True)

    return grouped

def timming_table_consolidation(df):

  # Melting the DataFrame to have 'rounds', 'strategy', 'variable', and 'value'
  df_melted = df.melt(id_vars=['strategy', 'rounds'], value_vars=['test_selection_time', 'test_training_time'])

  df_melted['value'] = df_melted['value'].round(1)

  # Create a pivot table with 'strategy' and 'variable' as rows (multi-index) and 'rounds' as columns
  pivot_table = df_melted.pivot_table(index=['strategy', 'variable'], columns='rounds', values='value', aggfunc='first')

  # Rename the 'variable' level with more descriptive names
  pivot_table.index.set_levels(['selection_time', 'training_time'], level=1, inplace=True)

  # Optionally, you can also reset the index to make 'strategy' and 'type' as regular columns
  pivot_table.reset_index(inplace=True)

  return pivot_table    