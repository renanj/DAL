
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
        if self.experiment_name == 'LRootV4_teste':
            self.data_set_name = 'LRootV4'
            self.experiment_name = 'LRootV4_teste'
            self.nclasses = 30
            self.custom_train_root = '/content/LRoot_sipi_v4_adjusted/train'
            self.custom_test_root = '/content/LRoot_sipi_v4_adjusted/test'


            self.initial_seed_size = 5000
            self.budget_list = [500]
            self.training_size_cap = 7000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 5, 'lr': 0.1, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                'random', 'margin'
            ]
                

        if self.experiment_name == 'LRootV4_Baseline':
            self.data_set_name = 'LRootV4'
            self.experiment_name = 'LRootV4_Baseline'
            self.nclasses = 30
            self.custom_train_root = '/content/LRoot_sipi_v4_adjusted/train'
            self.custom_test_root = '/content/LRoot_sipi_v4_adjusted/test'


            self.initial_seed_size = 5000
            self.budget_list = [3000]
            self.training_size_cap = 20000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'LRootV4_Baseline_DataAugumentation':
            self.data_set_name = 'LRootV4'
            self.experiment_name = 'LRootV4_Baseline_DataAugumentation'
            self.nclasses = 30
            self.custom_train_root = '/content/LRoot_sipi_v4_adjusted/train'
            self.custom_test_root = '/content/LRoot_sipi_v4_adjusted/test'


            self.initial_seed_size = 5000
            self.budget_list = [3000]
            self.training_size_cap = 20000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = True
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
                
        if self.experiment_name == 'LRootV4_Baseline_LowData':
            self.data_set_name = 'LRootV4'
            self.experiment_name = 'LRootV4_Baseline_LowData'
            self.nclasses = 30
            self.custom_train_root = '/content/LRoot_sipi_v4_adjusted/train'
            self.custom_test_root = '/content/LRoot_sipi_v4_adjusted/test'


            self.initial_seed_size = 3000
            self.budget_list = [200]
            self.training_size_cap = 4000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'LRootV4_Baseline_LowData_DataAugumentation':
            self.data_set_name = 'LRootV4'
            self.experiment_name = 'LRootV4_Baseline_LowData_DataAugumentation'
            self.nclasses = 30
            self.custom_train_root = '/content/LRoot_sipi_v4_adjusted/train'
            self.custom_test_root = '/content/LRoot_sipi_v4_adjusted/test'


            self.initial_seed_size = 3000
            self.budget_list = [200]
            self.training_size_cap = 4000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'LRootV4_LRootV4_BatchSize1000':
            self.data_set_name = 'LRootV4'
            self.experiment_name = 'LRootV4_LRootV4_BatchSize1000'
            self.nclasses = 30
            self.custom_train_root = '/content/LRoot_sipi_v4_adjusted/train'
            self.custom_test_root = '/content/LRoot_sipi_v4_adjusted/test'


            self.initial_seed_size = 5000
            self.budget_list = [1000]
            self.training_size_cap = 20000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'LRootV4_LRootV4_BatchSize3000':
            self.data_set_name = 'LRootV4'
            self.experiment_name = 'LRootV4_LRootV4_BatchSize3000'
            self.nclasses = 30
            self.custom_train_root = '/content/LRoot_sipi_v4_adjusted/train'
            self.custom_test_root = '/content/LRoot_sipi_v4_adjusted/test'


            self.initial_seed_size = 5000
            self.budget_list = [3000]
            self.training_size_cap = 20000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'LRootV4_LRootV4_BatchSize6000':
            self.data_set_name = 'LRootV4'
            self.experiment_name = 'LRootV4_LRootV4_BatchSize6000'
            self.nclasses = 30
            self.custom_train_root = '/content/LRoot_sipi_v4_adjusted/train'
            self.custom_test_root = '/content/LRoot_sipi_v4_adjusted/test'


            self.initial_seed_size = 5000
            self.budget_list = [6000]
            self.training_size_cap = 20000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'LRootV4_LRootV4_LowData_BatchSize100':
            self.data_set_name = 'LRootV4'
            self.experiment_name = 'LRootV4_LRootV4_LowData_BatchSize100'
            self.nclasses = 30
            self.custom_train_root = '/content/LRoot_sipi_v4_adjusted/train'
            self.custom_test_root = '/content/LRoot_sipi_v4_adjusted/test'


            self.initial_seed_size = 5000
            self.budget_list = [100]
            self.training_size_cap = 7000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'LRootV4_LRootV4_LowData_BatchSize250':
            self.data_set_name = 'LRootV4'
            self.experiment_name = 'LRootV4_LRootV4_LowData_BatchSize250'
            self.nclasses = 30
            self.custom_train_root = '/content/LRoot_sipi_v4_adjusted/train'
            self.custom_test_root = '/content/LRoot_sipi_v4_adjusted/test'


            self.initial_seed_size = 5000
            self.budget_list = [250]
            self.training_size_cap = 7000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'LRootV4_LRootV4_LowData_BatchSize500':
            self.data_set_name = 'LRootV4'
            self.experiment_name = 'LRootV4_LRootV4_LowData_BatchSize500'
            self.nclasses = 30
            self.custom_train_root = '/content/LRoot_sipi_v4_adjusted/train'
            self.custom_test_root = '/content/LRoot_sipi_v4_adjusted/test'


            self.initial_seed_size = 5000
            self.budget_list = [500]
            self.training_size_cap = 7000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'LRootV4_LRootV4_Epochs10':
            self.data_set_name = 'LRootV4'
            self.experiment_name = 'LRootV4_LRootV4_Epochs10'
            self.nclasses = 30
            self.custom_train_root = '/content/LRoot_sipi_v4_adjusted/train'
            self.custom_test_root = '/content/LRoot_sipi_v4_adjusted/test'


            self.initial_seed_size = 5000
            self.budget_list = [1000]
            self.training_size_cap = 20000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'LRootV4_LRootV4_Epochs50':
            self.data_set_name = 'LRootV4'
            self.experiment_name = 'LRootV4_LRootV4_Epochs50'
            self.nclasses = 30
            self.custom_train_root = '/content/LRoot_sipi_v4_adjusted/train'
            self.custom_test_root = '/content/LRoot_sipi_v4_adjusted/test'


            self.initial_seed_size = 5000
            self.budget_list = [1000]
            self.training_size_cap = 20000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 50, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'LRootV4_LRootV4_Epochs100':
            self.data_set_name = 'LRootV4'
            self.experiment_name = 'LRootV4_LRootV4_Epochs100'
            self.nclasses = 30
            self.custom_train_root = '/content/LRoot_sipi_v4_adjusted/train'
            self.custom_test_root = '/content/LRoot_sipi_v4_adjusted/test'


            self.initial_seed_size = 5000
            self.budget_list = [1000]
            self.training_size_cap = 20000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 100, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'ASLO_Baseline':
            self.data_set_name = 'ASLO'
            self.experiment_name = 'ASLO_Baseline'
            self.nclasses = 22
            self.custom_train_root = '/content/ASLO/trainning'
            self.custom_test_root = '/content/ASLO/validation'


            self.initial_seed_size = 300
            self.budget_list = [300]
            self.training_size_cap = 3300
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'ASLO_Baseline_DataAugumentation':
            self.data_set_name = 'ASLO'
            self.experiment_name = 'ASLO_Baseline_DataAugumentation'
            self.nclasses = 22
            self.custom_train_root = '/content/ASLO/trainning'
            self.custom_test_root = '/content/ASLO/validation'


            self.initial_seed_size = 300
            self.budget_list = [300]
            self.training_size_cap = 3300
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = True
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
                
        if self.experiment_name == 'ASLO_Baseline_LowData':
            self.data_set_name = 'ASLO'
            self.experiment_name = 'ASLO_Baseline_LowData'
            self.nclasses = 22
            self.custom_train_root = '/content/ASLO/trainning'
            self.custom_test_root = '/content/ASLO/validation'


            self.initial_seed_size = 50
            self.budget_list = [50]
            self.training_size_cap = 400
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'ASLO_Baseline_LowData_DataAugumentation':
            self.data_set_name = 'ASLO'
            self.experiment_name = 'ASLO_Baseline_LowData_DataAugumentation'
            self.nclasses = 22
            self.custom_train_root = '/content/ASLO/trainning'
            self.custom_test_root = '/content/ASLO/validation'


            self.initial_seed_size = 50
            self.budget_list = [50]
            self.training_size_cap = 400
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'ASLO_LRootV4_BatchSize100':
            self.data_set_name = 'ASLO'
            self.experiment_name = 'ASLO_LRootV4_BatchSize100'
            self.nclasses = 22
            self.custom_train_root = '/content/ASLO/trainning'
            self.custom_test_root = '/content/ASLO/validation'


            self.initial_seed_size = 300
            self.budget_list = [100]
            self.training_size_cap = 3300
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'ASLO_LRootV4_BatchSize300':
            self.data_set_name = 'ASLO'
            self.experiment_name = 'ASLO_LRootV4_BatchSize300'
            self.nclasses = 22
            self.custom_train_root = '/content/ASLO/trainning'
            self.custom_test_root = '/content/ASLO/validation'


            self.initial_seed_size = 300
            self.budget_list = [300]
            self.training_size_cap = 3300
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'ASLO_LRootV4_BatchSize500':
            self.data_set_name = 'ASLO'
            self.experiment_name = 'ASLO_LRootV4_BatchSize500'
            self.nclasses = 22
            self.custom_train_root = '/content/ASLO/trainning'
            self.custom_test_root = '/content/ASLO/validation'


            self.initial_seed_size = 300
            self.budget_list = [500]
            self.training_size_cap = 3300
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'ASLO_LRootV4_LowData_BatchSize5':
            self.data_set_name = 'ASLO'
            self.experiment_name = 'ASLO_LRootV4_LowData_BatchSize5'
            self.nclasses = 22
            self.custom_train_root = '/content/ASLO/trainning'
            self.custom_test_root = '/content/ASLO/validation'


            self.initial_seed_size = 50
            self.budget_list = [5]
            self.training_size_cap = 400
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'ASLO_LRootV4_LowData_BatchSize10':
            self.data_set_name = 'ASLO'
            self.experiment_name = 'ASLO_LRootV4_LowData_BatchSize10'
            self.nclasses = 22
            self.custom_train_root = '/content/ASLO/trainning'
            self.custom_test_root = '/content/ASLO/validation'


            self.initial_seed_size = 50
            self.budget_list = [10]
            self.training_size_cap = 400
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'ASLO_LRootV4_LowData_BatchSize25':
            self.data_set_name = 'ASLO'
            self.experiment_name = 'ASLO_LRootV4_LowData_BatchSize25'
            self.nclasses = 22
            self.custom_train_root = '/content/ASLO/trainning'
            self.custom_test_root = '/content/ASLO/validation'


            self.initial_seed_size = 50
            self.budget_list = [25]
            self.training_size_cap = 400
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'ASLO_LRootV4_Epochs10':
            self.data_set_name = 'ASLO'
            self.experiment_name = 'ASLO_LRootV4_Epochs10'
            self.nclasses = 22
            self.custom_train_root = '/content/ASLO/trainning'
            self.custom_test_root = '/content/ASLO/validation'


            self.initial_seed_size = 300
            self.budget_list = [300]
            self.training_size_cap = 3300
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'ASLO_LRootV4_Epochs50':
            self.data_set_name = 'ASLO'
            self.experiment_name = 'ASLO_LRootV4_Epochs50'
            self.nclasses = 22
            self.custom_train_root = '/content/ASLO/trainning'
            self.custom_test_root = '/content/ASLO/validation'


            self.initial_seed_size = 300
            self.budget_list = [300]
            self.training_size_cap = 3300
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 50, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'ASLO_LRootV4_Epochs100':
            self.data_set_name = 'ASLO'
            self.experiment_name = 'ASLO_LRootV4_Epochs100'
            self.nclasses = 22
            self.custom_train_root = '/content/ASLO/trainning'
            self.custom_test_root = '/content/ASLO/validation'


            self.initial_seed_size = 300
            self.budget_list = [300]
            self.training_size_cap = 3300
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 100, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'MNIST_Baseline':
            self.data_set_name = 'MNIST'
            self.experiment_name = 'MNIST_Baseline'
            self.nclasses = 10
            self.custom_train_root = '-'
            self.custom_test_root = '-'


            self.initial_seed_size = 1000
            self.budget_list = [3000]
            self.training_size_cap = 50000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'MNIST_Baseline_DataAugumentation':
            self.data_set_name = 'MNIST'
            self.experiment_name = 'MNIST_Baseline_DataAugumentation'
            self.nclasses = 10
            self.custom_train_root = '-'
            self.custom_test_root = '-'


            self.initial_seed_size = 1000
            self.budget_list = [3000]
            self.training_size_cap = 50000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = True
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
                
        if self.experiment_name == 'MNIST_Baseline_LowData':
            self.data_set_name = 'MNIST'
            self.experiment_name = 'MNIST_Baseline_LowData'
            self.nclasses = 10
            self.custom_train_root = '-'
            self.custom_test_root = '-'


            self.initial_seed_size = 50
            self.budget_list = [10]
            self.training_size_cap = 300
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'MNIST_Baseline_LowData_DataAugumentation':
            self.data_set_name = 'MNIST'
            self.experiment_name = 'MNIST_Baseline_LowData_DataAugumentation'
            self.nclasses = 10
            self.custom_train_root = '-'
            self.custom_test_root = '-'


            self.initial_seed_size = 50
            self.budget_list = [10]
            self.training_size_cap = 300
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'MNIST_LRootV4_BatchSize1000':
            self.data_set_name = 'MNIST'
            self.experiment_name = 'MNIST_LRootV4_BatchSize1000'
            self.nclasses = 10
            self.custom_train_root = '-'
            self.custom_test_root = '-'


            self.initial_seed_size = 1000
            self.budget_list = [1000]
            self.training_size_cap = 20000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'MNIST_LRootV4_BatchSize3000':
            self.data_set_name = 'MNIST'
            self.experiment_name = 'MNIST_LRootV4_BatchSize3000'
            self.nclasses = 10
            self.custom_train_root = '-'
            self.custom_test_root = '-'


            self.initial_seed_size = 1000
            self.budget_list = [3000]
            self.training_size_cap = 20000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'MNIST_LRootV4_BatchSize6000':
            self.data_set_name = 'MNIST'
            self.experiment_name = 'MNIST_LRootV4_BatchSize6000'
            self.nclasses = 10
            self.custom_train_root = '-'
            self.custom_test_root = '-'


            self.initial_seed_size = 1000
            self.budget_list = [6000]
            self.training_size_cap = 20000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'MNIST_LRootV4_LowData_BatchSize1':
            self.data_set_name = 'MNIST'
            self.experiment_name = 'MNIST_LRootV4_LowData_BatchSize1'
            self.nclasses = 10
            self.custom_train_root = '-'
            self.custom_test_root = '-'


            self.initial_seed_size = 50
            self.budget_list = [1]
            self.training_size_cap = 300
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'MNIST_LRootV4_LowData_BatchSize10':
            self.data_set_name = 'MNIST'
            self.experiment_name = 'MNIST_LRootV4_LowData_BatchSize10'
            self.nclasses = 10
            self.custom_train_root = '-'
            self.custom_test_root = '-'


            self.initial_seed_size = 50
            self.budget_list = [10]
            self.training_size_cap = 300
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'MNIST_LRootV4_LowData_BatchSize25':
            self.data_set_name = 'MNIST'
            self.experiment_name = 'MNIST_LRootV4_LowData_BatchSize25'
            self.nclasses = 10
            self.custom_train_root = '-'
            self.custom_test_root = '-'


            self.initial_seed_size = 50
            self.budget_list = [25]
            self.training_size_cap = 300
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'MNIST_LRootV4_Epochs10':
            self.data_set_name = 'MNIST'
            self.experiment_name = 'MNIST_LRootV4_Epochs10'
            self.nclasses = 10
            self.custom_train_root = '-'
            self.custom_test_root = '-'


            self.initial_seed_size = 1000
            self.budget_list = [3000]
            self.training_size_cap = 15000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'MNIST_LRootV4_Epochs50':
            self.data_set_name = 'MNIST'
            self.experiment_name = 'MNIST_LRootV4_Epochs50'
            self.nclasses = 10
            self.custom_train_root = '-'
            self.custom_test_root = '-'


            self.initial_seed_size = 1000
            self.budget_list = [3000]
            self.training_size_cap = 15000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 50, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                
        if self.experiment_name == 'MNIST_LRootV4_Epochs100':
            self.data_set_name = 'MNIST'
            self.experiment_name = 'MNIST_LRootV4_Epochs100'
            self.nclasses = 10
            self.custom_train_root = '-'
            self.custom_test_root = '-'


            self.initial_seed_size = 1000
            self.budget_list = [3000]
            self.training_size_cap = 15000
            self.model = 'Custom_VGG11'
            self.model_freeze_method = 'pre_trained_unfreeze_top_layer'


            self.args = [{'n_epoch': 100, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'freeze_method': 'pre_trained_unfreeze_top_layer', 'islogs': True, 'isverbose': True, 'device': 'cuda',  'isreset': True}]


            self.data_augumentation = False
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
                    

