from sklearn.model_selection import train_test_split
import pandas as pd
import random
import nltk
import re


class datahandler():
	def __init__(self, random_seed, unique = False, validation = False, is_tsv = False):
		super(datahandler, self).__init__()
		self.random_seed = random_seed
		self.validation = validation
		self.unique = unique
		self.is_tsv = is_tsv

		random.seed(self.random_seed)

	def load_dataset(self, path_dataset, task):# Loads the CSV files from the datasets [index: 0 = train, 1 = test, 2 = validation]
		if self.is_tsv:
			train = pd.read_csv(path_dataset + "train.tsv", sep='\t', encoding='UTF-8')
			test = pd.read_csv(path_dataset + "test.tsv", sep='\t', encoding='UTF-8')
		else:
			train = pd.read_csv(path_dataset + "train.csv", encoding = 'UTF-8')
			test  = pd.read_csv(path_dataset + "test.csv", encoding = 'UTF-8')
		if task == "A":
			train = train[train['task_1'].notna()]
			test = test[test['task_1'].notna()]
		elif task == "B":
			train = train[train['task_2'].notna()]
			test = test[test['task_2'].notna()]
		if self.validation == True:
			if self.is_tsv:
				train = pd.read_csv(path_dataset + "validation.tsv", sep='\t', encoding='UTF-8')
			else:
				validation = pd.read_csv(path_dataset + "validation.csv", encoding='UTF-8')
			if task == "A":
				validation = validation[validation['task_1'].notna()]
			elif task == "B":
				validation = validation[validation['task_2'].notna()]
			return [train, test, validation]
		return [train, test]

	def clean_sentence(self, sentence):
		# Convert instance to string
		sentence = str(sentence)

		# All text to lowecase
		sentence = sentence.lower()

		# Normalize users and url
		sentence = re.sub(r'\@\w+','@user', sentence)
		sentence = re.sub(r"http\S+|www\S+|https\S+", 'url', sentence, flags=re.MULTILINE)
		
		# Separate special characters
		sentence = re.sub(r":", " : ", sentence)
		sentence = re.sub(r",", " , ", sentence)
		sentence = re.sub(r"\.", " . ", sentence)
		sentence = re.sub(r"!", " ! ", sentence)
		sentence = re.sub(r"¡", " ¡ ", sentence)
		sentence = re.sub(r"“", " “ ", sentence)
		sentence = re.sub(r"”", " ” ", sentence)
		sentence = re.sub(r"\(", " ( ", sentence)
		sentence = re.sub(r"\)", " ) ", sentence)
		sentence = re.sub(r"\?", " ? ", sentence)
		sentence = re.sub(r"\¿", " ¿ ", sentence)

		# Substituting multiple spaces with single space
		sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)

		return sentence

	def clean_dataset(self, dataset, task):
		x_dataset = []
		y_dataset = []
		for i, row in dataset.iterrows():
			if row['text'].strip() != '':
				sentence = self.clean_sentence(row['text'])
				if task == "A":
					label = row['task_1']
				elif task == "B":
					label = row['task_2']
				x_dataset.append(sentence)
				y_dataset.append(label)
		return x_dataset, y_dataset
	
	def get_labels(self, dataset, task):
		y_dataset = []
		for i, row in dataset.iterrows():
			if row['text'].strip() != '':
				label = row[task]
				y_dataset.append(label)
		return y_dataset

	def preprocess_dataset(self, dataset, task):
		x_train, y_train = self.clean_dataset(dataset[0], task)
		x_test, y_test = self.clean_dataset(dataset[1], task)
		if self.validation == True:
			x_dev, y_dev = self.clean_dataset(dataset[2], task)
			return x_train, y_train, x_test, y_test, x_dev, y_dev
		else:
			return x_train, y_train, x_test, y_test