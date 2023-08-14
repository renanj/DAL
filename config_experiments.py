
import pandas as pd
import os
from torchvision import transforms

class ConfigExperiments:

  def __init__(self, experiment_name):

  	# Your initial attributes
  	self.logs_directory = '/content/drive/MyDrive/Colab_Notebooks/Experimentos_Mestrado/colab_storage/logs/'
  	self.checkpoint_directory = '/content/drive/MyDrive/Colab_Notebooks/Experimentos_Mestrado/colab_storage/check/'
  	self.model_directory = "/content/drive/MyDrive/Colab_Notebooks/Experimentos_Mestrado/colab_storage/model/"
  	self.dict_directory = "/content/drive/MyDrive/Colab_Notebooks/Experimentos_Mestrado/colab_storage/dict/"
    
  	# Generated blocks    
    if self.experiment_name == 'LRootV4_931':
      self.data_set_name = 'LRootV4'
      self.nclasses = 30
      self.experiment_name = 'LRootV4_Baseline'
      self.initial_seed_size = 5000
      self.budget_list = [3000]
      self.training_size_cap = 20000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'LRootV4_166':
      self.data_set_name = 'LRootV4'
      self.nclasses = 30
      self.experiment_name = 'LRootV4_Baseline_DataAugumentation'
      self.initial_seed_size = 5000
      self.budget_list = [3000]
      self.training_size_cap = 20000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = True
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'LRootV4_501':
      self.data_set_name = 'LRootV4'
      self.nclasses = 30
      self.experiment_name = 'LRootV4_Baseline_LowData'
      self.initial_seed_size = 3000
      self.budget_list = [200]
      self.training_size_cap = 4000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'LRootV4_633':
      self.data_set_name = 'LRootV4'
      self.nclasses = 30
      self.experiment_name = 'LRootV4_Baseline_LowData_DataAugumentation'
      self.initial_seed_size = 3000
      self.budget_list = [200]
      self.training_size_cap = 4000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'LRootV4_990':
      self.data_set_name = 'LRootV4'
      self.nclasses = 30
      self.experiment_name = 'LRootV4_LRootV4_BatchSize1000'
      self.initial_seed_size = 5000
      self.budget_list = [1000]
      self.training_size_cap = 20000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'LRootV4_916':
      self.data_set_name = 'LRootV4'
      self.nclasses = 30
      self.experiment_name = 'LRootV4_LRootV4_BatchSize3000'
      self.initial_seed_size = 5000
      self.budget_list = [3000]
      self.training_size_cap = 20000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'LRootV4_823':
      self.data_set_name = 'LRootV4'
      self.nclasses = 30
      self.experiment_name = 'LRootV4_LRootV4_BatchSize6000'
      self.initial_seed_size = 5000
      self.budget_list = [6000]
      self.training_size_cap = 20000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'LRootV4_442':
      self.data_set_name = 'LRootV4'
      self.nclasses = 30
      self.experiment_name = 'LRootV4_LRootV4_LowData_BatchSize100'
      self.initial_seed_size = 5000
      self.budget_list = [100]
      self.training_size_cap = 7000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'LRootV4_920':
      self.data_set_name = 'LRootV4'
      self.nclasses = 30
      self.experiment_name = 'LRootV4_LRootV4_LowData_BatchSize250'
      self.initial_seed_size = 5000
      self.budget_list = [250]
      self.training_size_cap = 7000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'LRootV4_909':
      self.data_set_name = 'LRootV4'
      self.nclasses = 30
      self.experiment_name = 'LRootV4_LRootV4_LowData_BatchSize500'
      self.initial_seed_size = 5000
      self.budget_list = [500]
      self.training_size_cap = 7000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'LRootV4_86':
      self.data_set_name = 'LRootV4'
      self.nclasses = 30
      self.experiment_name = 'LRootV4_LRootV4_Epochs10'
      self.initial_seed_size = 5000
      self.budget_list = [1000]
      self.training_size_cap = 20000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'LRootV4_802':
      self.data_set_name = 'LRootV4'
      self.nclasses = 30
      self.experiment_name = 'LRootV4_LRootV4_Epochs50'
      self.initial_seed_size = 5000
      self.budget_list = [1000]
      self.training_size_cap = 20000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 50, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'LRootV4_231':
      self.data_set_name = 'LRootV4'
      self.nclasses = 30
      self.experiment_name = 'LRootV4_LRootV4_Epochs100'
      self.initial_seed_size = 5000
      self.budget_list = [1000]
      self.training_size_cap = 20000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 100, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'ASLO_737':
      self.data_set_name = 'ASLO'
      self.nclasses = 22
      self.experiment_name = 'ASLO_Baseline'
      self.initial_seed_size = 300
      self.budget_list = [300]
      self.training_size_cap = 3300
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'ASLO_467':
      self.data_set_name = 'ASLO'
      self.nclasses = 22
      self.experiment_name = 'ASLO_Baseline_DataAugumentation'
      self.initial_seed_size = 300
      self.budget_list = [300]
      self.training_size_cap = 3300
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = True
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'ASLO_827':
      self.data_set_name = 'ASLO'
      self.nclasses = 22
      self.experiment_name = 'ASLO_Baseline_LowData'
      self.initial_seed_size = 50
      self.budget_list = [50]
      self.training_size_cap = 400
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'ASLO_777':
      self.data_set_name = 'ASLO'
      self.nclasses = 22
      self.experiment_name = 'ASLO_Baseline_LowData_DataAugumentation'
      self.initial_seed_size = 50
      self.budget_list = [50]
      self.training_size_cap = 400
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'ASLO_79':
      self.data_set_name = 'ASLO'
      self.nclasses = 22
      self.experiment_name = 'ASLO_LRootV4_BatchSize1000'
      self.initial_seed_size = 300
      self.budget_list = [100]
      self.training_size_cap = 3300
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'ASLO_790':
      self.data_set_name = 'ASLO'
      self.nclasses = 22
      self.experiment_name = 'ASLO_LRootV4_BatchSize3000'
      self.initial_seed_size = 300
      self.budget_list = [300]
      self.training_size_cap = 3300
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'ASLO_336':
      self.data_set_name = 'ASLO'
      self.nclasses = 22
      self.experiment_name = 'ASLO_LRootV4_BatchSize6000'
      self.initial_seed_size = 300
      self.budget_list = [500]
      self.training_size_cap = 3300
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'ASLO_518':
      self.data_set_name = 'ASLO'
      self.nclasses = 22
      self.experiment_name = 'ASLO_LRootV4_LowData_BatchSize100'
      self.initial_seed_size = 50
      self.budget_list = [5]
      self.training_size_cap = 400
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'ASLO_139':
      self.data_set_name = 'ASLO'
      self.nclasses = 22
      self.experiment_name = 'ASLO_LRootV4_LowData_BatchSize250'
      self.initial_seed_size = 50
      self.budget_list = [10]
      self.training_size_cap = 400
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'ASLO_710':
      self.data_set_name = 'ASLO'
      self.nclasses = 22
      self.experiment_name = 'ASLO_LRootV4_LowData_BatchSize500'
      self.initial_seed_size = 50
      self.budget_list = [25]
      self.training_size_cap = 400
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'ASLO_760':
      self.data_set_name = 'ASLO'
      self.nclasses = 22
      self.experiment_name = 'ASLO_LRootV4_Epochs10'
      self.initial_seed_size = 300
      self.budget_list = [300]
      self.training_size_cap = 3300
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'ASLO_772':
      self.data_set_name = 'ASLO'
      self.nclasses = 22
      self.experiment_name = 'ASLO_LRootV4_Epochs50'
      self.initial_seed_size = 300
      self.budget_list = [300]
      self.training_size_cap = 3300
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 50, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'ASLO_440':
      self.data_set_name = 'ASLO'
      self.nclasses = 22
      self.experiment_name = 'ASLO_LRootV4_Epochs100'
      self.initial_seed_size = 300
      self.budget_list = [300]
      self.training_size_cap = 3300
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 100, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'MNIST_579':
      self.data_set_name = 'MNIST'
      self.nclasses = 10
      self.experiment_name = 'MNIST_Baseline'
      self.initial_seed_size = 1000
      self.budget_list = [3000]
      self.training_size_cap = 50000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'MNIST_772':
      self.data_set_name = 'MNIST'
      self.nclasses = 10
      self.experiment_name = 'MNIST_Baseline_DataAugumentation'
      self.initial_seed_size = 1000
      self.budget_list = [3000]
      self.training_size_cap = 50000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = True
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'MNIST_790':
      self.data_set_name = 'MNIST'
      self.nclasses = 10
      self.experiment_name = 'MNIST_Baseline_LowData'
      self.initial_seed_size = 50
      self.budget_list = [10]
      self.training_size_cap = 300
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'MNIST_345':
      self.data_set_name = 'MNIST'
      self.nclasses = 10
      self.experiment_name = 'MNIST_Baseline_LowData_DataAugumentation'
      self.initial_seed_size = 50
      self.budget_list = [10]
      self.training_size_cap = 300
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'MNIST_461':
      self.data_set_name = 'MNIST'
      self.nclasses = 10
      self.experiment_name = 'MNIST_LRootV4_BatchSize1000'
      self.initial_seed_size = 1000
      self.budget_list = [1000]
      self.training_size_cap = 20000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'MNIST_48':
      self.data_set_name = 'MNIST'
      self.nclasses = 10
      self.experiment_name = 'MNIST_LRootV4_BatchSize3000'
      self.initial_seed_size = 1000
      self.budget_list = [3000]
      self.training_size_cap = 20000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'MNIST_610':
      self.data_set_name = 'MNIST'
      self.nclasses = 10
      self.experiment_name = 'MNIST_LRootV4_BatchSize6000'
      self.initial_seed_size = 1000
      self.budget_list = [6000]
      self.training_size_cap = 20000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'MNIST_91':
      self.data_set_name = 'MNIST'
      self.nclasses = 10
      self.experiment_name = 'MNIST_LRootV4_LowData_BatchSize1'
      self.initial_seed_size = 50
      self.budget_list = [1]
      self.training_size_cap = 300
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'MNIST_269':
      self.data_set_name = 'MNIST'
      self.nclasses = 10
      self.experiment_name = 'MNIST_LRootV4_LowData_BatchSize10'
      self.initial_seed_size = 50
      self.budget_list = [10]
      self.training_size_cap = 300
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'MNIST_372':
      self.data_set_name = 'MNIST'
      self.nclasses = 10
      self.experiment_name = 'MNIST_LRootV4_LowData_BatchSize25'
      self.initial_seed_size = 50
      self.budget_list = [25]
      self.training_size_cap = 300
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'MNIST_243':
      self.data_set_name = 'MNIST'
      self.nclasses = 10
      self.experiment_name = 'MNIST_LRootV4_Epochs10'
      self.initial_seed_size = 1000
      self.budget_list = [3000]
      self.training_size_cap = 15000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 10, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'MNIST_710':
      self.data_set_name = 'MNIST'
      self.nclasses = 10
      self.experiment_name = 'MNIST_LRootV4_Epochs50'
      self.initial_seed_size = 1000
      self.budget_list = [3000]
      self.training_size_cap = 15000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 50, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
          
    if self.experiment_name == 'MNIST_844':
      self.data_set_name = 'MNIST'
      self.nclasses = 10
      self.experiment_name = 'MNIST_LRootV4_Epochs100'
      self.initial_seed_size = 1000
      self.budget_list = [3000]
      self.training_size_cap = 15000
      self.model = 'Custom_VGG11'
      self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
      self.fine_tuning = True
      self.data_augumentation = False
      self.args = [{'n_epoch': 100, 'lr': 0.01, 'batch_size': 40, 'max_accuracy': 0.98, 'isreset': True}]
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
              'random', 'least_confidence', 'margin', 'entropy', 'badge', 'coreset', 
              'bald', 'glister', 'fass', 'batch_bald', 'adversarial_bim', 'adversarial_deepfool'
          ]
              

