import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import os



import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, ConnectionPatch



import os
import pickle
import pandas as pd
import json


##### Dataframe

def find_paths(directory):
    return [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def find_subdirectories_names(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]



def process_experiment_path(path):
    split_path = path.split('/')
    _experiment = split_path[-1]
    database_name = split_path[-2]  # Extracting the database name from the path

    _list_paths = find_paths(path)
    _list_strategies = find_subdirectories_names(path)

    _list_dfs = []
    error_paths = []  # To store paths where files couldn't be opened

    for i in range(len(_list_paths)):
        try:
            pkl_file_name = os.path.join(_list_paths[i], 'logs_dict.pkl')
            json_file_name = os.path.join(_list_paths[i], 'logs_dict.json')

            # Check which file exists and load accordingly
            if os.path.exists(pkl_file_name):
                with open(pkl_file_name, 'rb') as handle:
                    dict_random = pickle.load(handle)
            elif os.path.exists(json_file_name):
                with open(json_file_name, 'r') as handle:
                    dict_random = json.load(handle)
            else:
                error_paths.append(_list_paths[i])  # If neither file exists, add to error paths
                continue  # Skip the current iteration as no valid file was found

            database = database_name
            strategy = _list_strategies[i]

            rounds = list(range(1, len(dict_random['test_training_time']) + 1))
            dict_list = [
                [database] * len(rounds),
                [_experiment] * len(rounds),
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

            df['rounds'] = df['rounds'].astype(int)
            df['len_training_points'] = df['len_training_points'].astype(int)
            df['test_accuracy'] = df['test_accuracy'].astype(float)
            df['test_selection_time'] = df['test_selection_time'].astype(float)
            df['test_training_time'] = df['test_training_time'].astype(float)

            _list_dfs.append(df)
        except Exception as e:
            print(f"Error encountered: {e}")  # This will print the specific error
            error_paths.append(_list_paths[i])  # Add the current path to error paths due to the exception

    if error_paths:  # Check if there were any errors
        print("Error paths:", error_paths)

    if _list_dfs:  # Check if the list is not empty
        return pd.concat(_list_dfs)
    else:
        return None  # Return None or an empty DataFrame if you prefer: pd.DataFrame()




def process_path(_path):
    dfs = []  # This will hold all the dataframes created


    # Split the path into its components
    components = _path.split('/')
    # Removing the empty strings from the split result
    components = [comp for comp in components if comp]

    # Count the number of subdirectories beyond 'dict'
    if 'dict' in components:
        subdirs_beyond_dict = len(components) - components.index('dict') - 1
    else:
        return "Path is not in the expected pattern."

    # Differentiate based on the count
    if subdirs_beyond_dict == 0:
      print("This is the dicts path.")
      for database_path in find_paths(_path):
          for experiment_path in find_paths(database_path):
              df = process_experiment_path(experiment_path)
              if df is not None:
                  dfs.append(df)
      return pd.concat(dfs) if dfs else None        

    elif subdirs_beyond_dict == 1:
      print("This is the database path.") 
      # Database path
      for experiment_path in find_paths(_path):
          df = process_experiment_path(experiment_path)
          if df is not None:
              dfs.append(df)
      return pd.concat(dfs) if dfs else None        


    elif subdirs_beyond_dict == 2:
        print("This is the experiment path.")
        df = process_experiment_path(_path)
        return df

    else:
      print("Path pattern is not recognized." )
      return None




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



##### Visualization


def plot_accuracy_over_interactions(df):

    _option_x = 'len_training_points'


    fig, ax1 = plt.subplots(figsize=(12, 8))
    query_strategies = df['strategy'].unique()

    # Define a color for each unique strategy, note that 'random' is specifically set to black
    color_dict = {strategy: color for strategy, color in zip(query_strategies, sns.color_palette('pastel', n_colors=len(query_strategies)))}
    color_dict['random'] = 'black'

    max_interactions_to_plot = df.groupby('strategy')[_option_x].max().max()  # Calculate the maximum interaction value for all strategies

    for query_strategy in query_strategies:
        data = df[df['strategy'] == query_strategy]
        linewidth = 3 if query_strategy == 'random' else 2.5  # Make 'random' strategy line slightly thicker
        ax1.plot(data[_option_x], data['test_accuracy'], label=query_strategy, color=color_dict[query_strategy], linewidth=linewidth)

    ax1.set_xlabel(_option_x, fontsize=14)
    ax1.set_ylabel('Accuracy (Validation)', fontsize=14)
    ax1.set_title('Accuracy over Interactions', fontsize=16, weight='bold')
    ax1.legend(frameon=False, fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    sns.set(style="whitegrid")

    ax1.set_xlim(1, max_interactions_to_plot)

    plt.tight_layout()
    plt.show()



def plot_with_arrow_to_zoom(df, zoom_regions=None, std_dev=None, colors_dict=None):
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.color': '.9'})
    fig, ax = plt.subplots(figsize=(10, 6))

    plt.rcParams["font.family"] = "serif"

    if colors_dict is None:
        colors_dict = dict(zip(df['strategy'].unique(), sns.color_palette(n_colors=len(df['strategy'].unique()))))

    sns.lineplot(data=df, x="len_training_points", y="test_accuracy", hue="strategy", palette=colors_dict, lw=2, marker="o", ax=ax)

    ax.set_title("Test Accuracy vs. Length of Training Points", fontsize=18)
    ax.set_xlabel("Length of Training Points", fontsize=15)
    ax.set_ylabel("Test Accuracy", fontsize=15)
    ax.tick_params(labelsize=13)

    ax.legend(title='Strategy', title_fontsize='14', fontsize='13')

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
            ax_zoom.set_xticks([])
            ax_zoom.set_yticks([])
            ax_zoom.set_xlabel('')
            ax_zoom.set_ylabel('')

            # Draw rectangle in the main plot
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='lightgray', linestyle='dashed')
            ax.add_patch(rect)

            # Add connection lines
            con1 = ConnectionPatch(xyA=(x1, y1), xyB=(pos[0], pos[1]), coordsA="data", coordsB="axes fraction", axesA=ax, axesB=ax, color='lightgray')
            con2 = ConnectionPatch(xyA=(x2, y1), xyB=(pos[0] + pos[2], pos[1]), coordsA="data", coordsB="axes fraction", axesA=ax, axesB=ax, color='lightgray')

            ax.add_artist(con1)
            ax.add_artist(con2)

    plt.tight_layout()
    plt.show()    