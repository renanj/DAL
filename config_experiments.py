import os

class ConfigExperiments:

	def __init__(self,experiment_name):

		self.logs_directory = '/content/drive/MyDrive/Colab_Notebooks/Experimentos_Mestrado/colab_storage/logs/'
		self.checkpoint_directory = '/content/drive/MyDrive/Colab_Notebooks/Experimentos_Mestrado/colab_storage/check/'
		self.model_directory = "/content/drive/MyDrive/Colab_Notebooks/Experimentos_Mestrado/colab_storage/model/"
		self.dict_directory = "/content/drive/MyDrive/Colab_Notebooks/Experimentos_Mestrado/colab_storage/dict/"


		if self.experiment_name == 'TBD':

			self.model = 'Custom_VGG_11'
			self.model_freeze_method = 'pre_trained_unfreeze_top_layer'
			#'from_scratch', 'pre_trained_unfreeze_top_layer', 'pre_trained_unfreeze_partial_last_layers', 'pre_trained_unfreeze_all_layers'


			self.data_set_name = "LRoot"
			self.experiment_name = "LRoot_Baseline_V2"
			self.custom_train_root = '/content/LRoot_sipi_v4_adjusted/train'
			self.custom_test_root = '/content/LRoot_sipi_v4_adjusted/test'

			self.nclasses = 30
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
			  'random'
			  ,'least_confidence'
			  ,'margin'
			  ,'entropy'
			  ,'badge'
			  ,'coreset'
			  ,'bald'
			  ,'glister'
			  ,'fass'
			  ,'batch_bald'
			  ,'adversarial_bim' 
			  ,'adversarial_deepfool'
			  ]


			self.initial_seed_size = 2000
			self.training_size_cap = 20000
			self.budget_list = [3000]
			self.n_rounds_list = [(training_size_cap - initial_seed_size) // budget for budget in budget_list]	
			self.args = {'n_epoch':300, 'lr':float(0.1), 'batch_size':20, 'max_accuracy':float(0.98), 'islogs':True, 'isreset':True, 'isverbose':True, 'device':'cuda'}					




		elif self.test_number == 'mnist_1':




		elif self.test_number == 'mnist_2':

	

		elif self.test_number == 'plancton_1':

	

		elif self.test_number == 'plancton_2':			



		else:
			raise ValueError("Invalid Experiment Name!")						

	
# Get the configuration name from the environment variable.
test_number = os.getenv('TEST_NUMBER')

if test_number is None:
    raise ValueError('TEST_NUMBER environment variable not set. Please provide a configuration name.')

# Create a Config instance with the appropriate configuration.
config = Config(test_number)
