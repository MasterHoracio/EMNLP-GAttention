# Avoid warnings from libraries
import warnings
warnings.filterwarnings('ignore')

from transformers import get_linear_schedule_with_warmup, logging
from torch.optim import AdamW
from sklearn.metrics import classification_report
from Utilities import texthandler
from Utilities import tokenhandler
from Utilities import architectures
from Utilities import trainhandler
from torch import nn
import pandas as pd
import numpy as np
import torch
import random
import copy
import os

logging.set_verbosity_error() # Remove warnings from the pre-trained models

class pipeline():
    def __init__(self, dataset_name, task, PRE_TRAINED_MODEL_NAME, architecture_mode, MAX_LENGTH, encoding_dimension, is_tsv = False, unique = False, validation = False):
        super(pipeline, self).__init__()
        self.random_seed                = self.get_random_seed_through_os()
        self.dataset_name               = dataset_name
        self.task                       = task
        self.PRE_TRAINED_MODEL_NAME     = PRE_TRAINED_MODEL_NAME
        self.architecture_mode          = architecture_mode
        self.MAX_LENGTH                 = MAX_LENGTH
        self.encoding_dimension         = encoding_dimension
        self.validation                 = validation
        self.unique                     = unique
        self.is_tsv                     = is_tsv

        print("****************************************************************")
        print("Training on: " + self.PRE_TRAINED_MODEL_NAME)
        print("Using dataset: " + self.dataset_name + " Task: " + self.task)
        print("****************************************************************")

    def get_random_seed_through_os(self):
        RAND_SIZE = 4
        random_data = os.urandom(RAND_SIZE)
        random_seed = int.from_bytes(random_data, byteorder="big")
        return random_seed

    def get_hyperparameters(self):
        if self.dataset_name == "SEM":
            path = "Datasets/semeval 2019 task 6/"
            if self.task == "A":
                #LABELS
                label_transform = lambda x: 1 if x == 'OFF' else 0  # Semeval 2019
                N_CLASSES = 2 # Define the number of classes
                labels = [0,1]
                target_names = ["NOT","OFF"]
                #NN HYPERPARAMETERS
                epochs = 2 # Define the number of epochs 3e-5
                BATCH_SIZE = 32 # Define the number of epochs 3e-5
                LEARNING_RATE = 4e-5
            elif self.task == "B":
                #LABELS
                label_transform = lambda x: 1 if x == 'TIN' else 0  # Semeval 2019
                N_CLASSES = 2 # Define the number of classes
                labels = [0,1]
                target_names = ["UNT","TIN"]
                #NN HYPERPARAMETERS
                epochs = 4 # Define the number of epochs 3e-5
                BATCH_SIZE = 16 # Define the number of epochs 3e-5
                LEARNING_RATE = 1.5e-5
        elif self.dataset_name == "AMI":
            path = "Datasets/ami 2018/"
            if self.task == "A":
                #LABELS
                label_transform = lambda x: 1 if x == 1 else 0  # AMI Evalita
                N_CLASSES = 2 # Define the number of classes
                labels = [0,1]
                target_names = ["NOT","MIS"]
                #NN HYPERPARAMETERS
                epochs = 3
                BATCH_SIZE = 18
                LEARNING_RATE = 2e-5
            elif self.task == "B":
                #LABELS
                label_transform = lambda x: 1 if x == 'dominance' else 2 if x == 'sexual_harassment' else 3 if x == 'derailing' else 4 if x == 'stereotype' else 5 if x == 'discredit' else 0  # AMI Evalita
                label_transform_2 = lambda x: 1 if x == 'active' else 2 if x == 'passive' else 0  # AMI Evalita
                N_CLASSES =  6# Define the number of classes
                N_CLASSES_2 =  3# Define the number of classes
                labels = [0,1,2,3,4,5]
                target_names = ["NOT","DOM","SEX","DER","STE","DIS"]
                labels_2 = [0,1,2]
                target_names_2 = ["NOT","PAS","ACT"]
                #NN HYPERPARAMETERS
                epochs = 3
                BATCH_SIZE = 12
                LEARNING_RATE = 2.5e-5
                return path, label_transform, label_transform_2, N_CLASSES, N_CLASSES_2, labels, target_names, labels_2, target_names_2, epochs, BATCH_SIZE, LEARNING_RATE
        elif self.dataset_name == "HAS":
            path = "Datasets/hasoc 2019/"
            if self.task == "A":
                #LABELS
                label_transform = lambda x: 1 if x == 'HOF' else 0  # HASOC 2019
                N_CLASSES = 2
                labels = [0,1]
                target_names = ["NOT","HOF"]
                #NN HYPERPARAMETERS
                epochs = 2
                BATCH_SIZE = 24
                LEARNING_RATE = 5e-5
            elif self.task == "B":
                #LABELS
                label_transform = lambda x: 1 if x == 'HATE' else 2 if x == 'PRFN' else 3 if x == 'OFFN' else 0 # HASOC 2019
                N_CLASSES = 4 # Define the number of classes
                labels = [0,1,2,3]
                target_names = ["NOT","HATE","PRFN","OFFN"]
                #NN HYPERPARAMETERS
                epochs = 4
                BATCH_SIZE = 24
                LEARNING_RATE = 1.5e-5
        return path, label_transform, N_CLASSES, labels, target_names, epochs, BATCH_SIZE, LEARNING_RATE
    
    # Normalize dataset labels
    def normalize_labels(self, labels, ts):
        transformed_labels = []
        for label in labels:
            transformed_labels.append(ts(label))
        return transformed_labels
    
    def get_full_dataset(self, path, label_transform, label_transform_2 = None):
        data_handler = texthandler.datahandler(self.random_seed, self.unique, self.validation, self.is_tsv)
        # Load dataset from CSV
        dataset = data_handler.load_dataset(path, self.task)

        # Pre-process dataset (clean text and split the dataset into train, test and validation)
        x_train, y_train, x_test, y_test = data_handler.preprocess_dataset(dataset, self.task)

        if self.dataset_name == "AMI" and self.task == "B":#Load the additional labels for the subtask B of AMI dataset
            y2_train = data_handler.get_labels(dataset[0], "task_3")
            y2_test = data_handler.get_labels(dataset[1], "task_3")

        y_train = self.normalize_labels(y_train,label_transform)
        y_test = self.normalize_labels(y_test, label_transform)

        if self.dataset_name == "AMI" and self.task == "B":
            y2_train = self.normalize_labels(y2_train, label_transform_2)
            y2_test = self.normalize_labels(y2_test, label_transform_2)
        else:
            y2_train = None
            y2_test = None

        return x_train, y_train, y2_train, x_test, y_test, y2_test
    
    def tokenize_full_dataset(self, x_train, y_train, y2_train, x_test, y_test, y2_test, BATCH_SIZE):
        # Create tokenizer instance
        tokenizer = tokenhandler.tokenhandler(self.PRE_TRAINED_MODEL_NAME, self.MAX_LENGTH)

        # Tokenize all the dataset
        if self.PRE_TRAINED_MODEL_NAME == "bert-base-uncased" or self.PRE_TRAINED_MODEL_NAME == "nghuyong/ernie-2.0-base-en":
            input_ids_train, input_token_ids_train, attention_masks_train = tokenizer.tokenize_dataset(x_train)
            input_ids_test, input_token_ids_test, attention_masks_test = tokenizer.tokenize_dataset(x_test)
        elif self.PRE_TRAINED_MODEL_NAME == "distilbert-base-uncased" or self.PRE_TRAINED_MODEL_NAME == "roberta-base":
            input_ids_train, attention_masks_train = tokenizer.tokenize_dataset(x_train)
            input_ids_test, attention_masks_test = tokenizer.tokenize_dataset(x_test)

        # Convert all the dataset into a single group of tensors
        if self.PRE_TRAINED_MODEL_NAME == "bert-base-uncased" or self.PRE_TRAINED_MODEL_NAME == "nghuyong/ernie-2.0-base-en":
            train_dataset = tokenizer.tensor_the_dataset_bert(input_ids_train, input_token_ids_train, attention_masks_train, y_train, y2_train)
            test_dataset = tokenizer.tensor_the_dataset_bert(input_ids_test, input_token_ids_test, attention_masks_test, y_test, y2_test)
        elif self.PRE_TRAINED_MODEL_NAME == "distilbert-base-uncased" or self.PRE_TRAINED_MODEL_NAME == "roberta-base":
            train_dataset = tokenizer.tensor_the_dataset_distil_bert(input_ids_train, attention_masks_train, y_train, y2_train)
            test_dataset = tokenizer.tensor_the_dataset_distil_bert(input_ids_test, attention_masks_test, y_test, y2_test)
        
        # We create the DataLoaders for training, testing and development
        train_dataloader = tokenizer.create_dataloader_random(train_dataset, BATCH_SIZE)
        test_dataloader = tokenizer.create_dataloader_sequential(test_dataset, BATCH_SIZE)

        return train_dataloader, test_dataloader
    
    def load_model(self, N_CLASSES, N_CLASSES_2 = None):
        # Assign the training device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.architecture_mode == "MHG":
            if self.PRE_TRAINED_MODEL_NAME == "bert-base-uncased" or self.PRE_TRAINED_MODEL_NAME == "nghuyong/ernie-2.0-base-en":
                if self.dataset_name == "AMI" and self.task == "B":
                    model = architectures.BERTMHGAttention(self.encoding_dimension, self.PRE_TRAINED_MODEL_NAME, self.MAX_LENGTH, n_classes = N_CLASSES, device = device, n_classes2 = N_CLASSES_2)
                else:
                    model = architectures.BERTMHGAttention(self.encoding_dimension, self.PRE_TRAINED_MODEL_NAME, self.MAX_LENGTH, n_classes = N_CLASSES, device = device)
            elif self.PRE_TRAINED_MODEL_NAME == "distilbert-base-uncased" or self.PRE_TRAINED_MODEL_NAME == "roberta-base":
                if self.dataset_name == "AMI" and self.task == "B":
                    model = architectures.DistilBERTMHGAttention(self.encoding_dimension, self.PRE_TRAINED_MODEL_NAME, self.MAX_LENGTH, n_classes = N_CLASSES, device = device, n_classes2 = N_CLASSES_2)
                else:
                    model = architectures.DistilBERTMHGAttention(self.encoding_dimension, self.PRE_TRAINED_MODEL_NAME, self.MAX_LENGTH, n_classes = N_CLASSES, device = device)
        elif self.architecture_mode == "GAT":
            if self.PRE_TRAINED_MODEL_NAME == "bert-base-uncased" or self.PRE_TRAINED_MODEL_NAME == "nghuyong/ernie-2.0-base-en":
                if self.dataset_name == "AMI" and self.task == "B":
                    model = architectures.BERTGAttention(self.encoding_dimension, self.PRE_TRAINED_MODEL_NAME, self.MAX_LENGTH, n_classes = N_CLASSES, device = device, n_classes2 = N_CLASSES_2)
                else:
                    model = architectures.BERTGAttention(self.encoding_dimension, self.PRE_TRAINED_MODEL_NAME, self.MAX_LENGTH, n_classes = N_CLASSES, device = device)
            elif self.PRE_TRAINED_MODEL_NAME == "distilbert-base-uncased" or self.PRE_TRAINED_MODEL_NAME == "roberta-base":
                if self.dataset_name == "AMI" and self.task == "B":
                    model = architectures.DistilBERTGAttention(self.encoding_dimension, self.PRE_TRAINED_MODEL_NAME, self.MAX_LENGTH, n_classes = N_CLASSES, device = device, n_classes2 = N_CLASSES_2)
                else:
                    model = architectures.DistilBERTGAttention(self.encoding_dimension, self.PRE_TRAINED_MODEL_NAME, self.MAX_LENGTH, n_classes = N_CLASSES, device = device)
        elif self.architecture_mode == "NON":
            if self.PRE_TRAINED_MODEL_NAME == "bert-base-uncased" or self.PRE_TRAINED_MODEL_NAME == "nghuyong/ernie-2.0-base-en":
                if self.dataset_name == "AMI" and self.task == "B":
                    model = architectures.BERT(self.encoding_dimension, self.PRE_TRAINED_MODEL_NAME, n_classes = N_CLASSES, n_classes2 = N_CLASSES_2)
                else:
                    model = architectures.BERT(self.encoding_dimension, self.PRE_TRAINED_MODEL_NAME, n_classes = N_CLASSES)
            elif self.PRE_TRAINED_MODEL_NAME == "distilbert-base-uncased" or self.PRE_TRAINED_MODEL_NAME == "roberta-base":
                if self.dataset_name == "AMI" and self.task == "B":
                    model = architectures.DistilBERT(self.encoding_dimension, self.PRE_TRAINED_MODEL_NAME, n_classes = N_CLASSES, n_classes2 = N_CLASSES_2)
                else:
                    model = architectures.DistilBERT(self.encoding_dimension, self.PRE_TRAINED_MODEL_NAME, n_classes = N_CLASSES)

        n_params = sum(p.numel() for p in model.parameters())

        return model, n_params, device

    def train_model(self, model, device, train_dataloader, epochs, LEARNING_RATE):
        # Load the model into the GPU
        model.to(device)

        # Define the optimization algorithm
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

        # Define the number of training steps (epochs * number of batches)
        total_steps = len(train_dataloader) * epochs

        # Create a schedule for the LR update
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

        # Define the loss function
        criterion = nn.CrossEntropyLoss()
        
        # Create trainhabnlder instance
        t_handler = trainhandler.trainhandler(device, scheduler)

        # Training cicle
        best_f1_score = 0
        for epoch in range(epochs):
            if self.PRE_TRAINED_MODEL_NAME == "bert-base-uncased" or self.PRE_TRAINED_MODEL_NAME == "nghuyong/ernie-2.0-base-en":
                if self.dataset_name == "AMI" and self.task == "B":
                    t_handler.train_bert(epoch + 1, epochs, model, train_dataloader, optimizer, criterion, scheduler, True)
                else:
                    t_handler.train_bert(epoch + 1, epochs, model, train_dataloader, optimizer, criterion, scheduler, False)
            elif self.PRE_TRAINED_MODEL_NAME == "distilbert-base-uncased" or self.PRE_TRAINED_MODEL_NAME == "roberta-base":
                if self.dataset_name == "AMI" and self.task == "B":
                    t_handler.train_distil_bert(epoch + 1, epochs, model, train_dataloader, optimizer, criterion, scheduler, True)
                else:
                    t_handler.train_distil_bert(epoch + 1, epochs, model, train_dataloader, optimizer, criterion, scheduler, False)
        
        return t_handler

    def evaluate(self, t_handler, model, test_dataloader):
        if self.PRE_TRAINED_MODEL_NAME == "bert-base-uncased" or self.PRE_TRAINED_MODEL_NAME == "nghuyong/ernie-2.0-base-en":
            if self.dataset_name == "AMI" and self.task == "B":
                f1_test, model_prediction_test, ground_truth_test, f1_test_2, model_prediction_test_2, ground_truth_test_2 = t_handler.evaluate_model_bert(model, test_dataloader, True)
            else:
                f1_test, model_prediction_test, ground_truth_test = t_handler.evaluate_model_bert(model, test_dataloader, False)
        elif self.PRE_TRAINED_MODEL_NAME == "distilbert-base-uncased" or self.PRE_TRAINED_MODEL_NAME == "roberta-base":
            if self.dataset_name == "AMI" and self.task == "B":
                f1_test, model_prediction_test, ground_truth_test, f1_test_2, model_prediction_test_2, ground_truth_test_2 = t_handler.evaluate_model_distil_bert(model, test_dataloader, True)
            else:
                f1_test, model_prediction_test, ground_truth_test = t_handler.evaluate_model_distil_bert(model, test_dataloader, False)

        if self.dataset_name == "AMI" and self.task == "B":
            return f1_test, model_prediction_test, ground_truth_test, f1_test_2, model_prediction_test_2, ground_truth_test_2
        else:
            return f1_test, model_prediction_test, ground_truth_test
    
    def report(self, ground_truth_test, model_prediction_test, labels, target_names, f1_test, save_labels, ground_truth_test_2 = None, model_prediction_test_2 = None, labels_2 = None, target_names_2 = None, f1_test_2 = None):
        if self.dataset_name == "AMI" and self.task == "B":
            print(classification_report(ground_truth_test, model_prediction_test, labels=labels, target_names=target_names, digits=4))
            print(classification_report(ground_truth_test_2, model_prediction_test_2, labels=labels_2, target_names=target_names_2, digits=4))
            print(f"Evaluation score:{(f1_test+f1_test_2)/2}")
        else:
            print(classification_report(ground_truth_test, model_prediction_test, labels=labels, target_names=target_names, digits=4))

        if save_labels == True:
            if self.dataset_name == "AMI" and self.task == "B":
                df = pd.DataFrame({'ground_truth_t1': ground_truth_test, 'predictions_t1': model_prediction_test, 'ground_truth_t2': ground_truth_test_2, 'predictions_t2': model_prediction_test_2})
            else:
                df = pd.DataFrame({'ground_truth': ground_truth_test, 'predictions': model_prediction_test})
            prefix = self.dataset_name + "_" + self.task + "_"
            df.to_csv(prefix+'predictions.csv', index=False)