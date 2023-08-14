import pandas as pd


def generate_if_blocks(df):

    code_blocks = []

    # Additional content you want to add inside each if block
    additional_content_inside_if = """
self.train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7928, 0.7928, 0.7928], std=[0.1688, 0.1688, 0.1688])
])

self.test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7928, 0.7928, 0.7928], std=[0.1688, 0.1688, 0.1688])
])

self.strategy_list = [
    'random', 'margin', 'badge', 'bald', 'batch_bald',        
    'glister', 'fass',
    'entropy', 'least_confidence', 'coreset', 
    'adversarial_bim', 'adversarial_deepfool'
]
    """

    for _, row in df.iterrows():

        block = []
        experiment_name = row['Experiment Name']
        
        block.append(f"        if self.experiment_name == '{experiment_name}':")
        block.append(f"            self.data_set_name = '{row['Dataset']}'")
        block.append(f"            self.experiment_name = '{row['Experiment Name']}'")
        block.append(f"            self.nclasses = {row['Classes']}")
        block.append(f"            self.custom_train_root = '{row['path_train']}'")
        block.append(f"            self.custom_test_root = '{row['path_test']}'")
        block.append('\n')        
        block.append(f"            self.initial_seed_size = {row['Initial Seed']}")
        block.append(f"            self.budget_list = [{row['Batch Size']}]")
        block.append(f"            self.training_size_cap = {row['Training Size Cap']}")
        block.append(f"            self.model = '{row['CNN Network']}'")
        block.append(f"            self.model_freeze_method = '{row['Freeze-Method']}'")
        # block.append(f"            self.fine_tuning = {row['fine_tuning']}")
        block.append('\n')
        block.append(f"            self.args = [{{'n_epoch': {row['epochs']}, 'lr': {float(row['Learning Rate'])}, 'batch_size': {row['batch_size']}, 'max_accuracy': {float(row['max accuracy'])}, 'freeze_method': '{row['Freeze-Method']}', 'islogs': {True}, 'isverbose': {True}, 'device': 'cuda',  'isreset': {row['is_reset (model)? ']} }}]")
        block.append('\n')
        block.append(f"            self.data_augumentation = {row['Data Agumentation']}")        
        # Add the additional content here
        block.extend(["            "+line for line in additional_content_inside_if.split('\n') if line])

        code_blocks.append('\n'.join(block))







    return '\n'.join(code_blocks)


df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRl43-tS1F_9hm6yTS-iUWsJP6nVAzdgLV1mogjP618a_FZnaq2DznxF9gzSyWAh1PQ9bK68SRyhTpY/pub?gid=1663178760&single=true&output=csv')                

blocks = generate_if_blocks(df)    

content = f"""
import pandas as pd
import os
from torchvision import transforms

class ConfigExperiments:

  def __init__(self, experiment_name):

        self.experiment_name = experiment_name
        # Your initial attributes
        self.logs_directory = '/content/drive/MyDrive/Colab_Notebooks/Experimentos_Mestrado/colab_storage/logs/'
        self.checkpoint_directory = '/content/drive/MyDrive/Colab_Notebooks/Experimentos_Mestrado/colab_storage/check/'
        self.model_directory = "/content/drive/MyDrive/Colab_Notebooks/Experimentos_Mestrado/colab_storage/model/"
        self.dict_directory = "/content/drive/MyDrive/Colab_Notebooks/Experimentos_Mestrado/colab_storage/dict/"

        # Generated blocks    
{blocks}    

"""



with open("config_experiments.py", "w") as f:
    f.write(content)