import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
import torch

class tokenhandler():
    def __init__(self, PRE_TRAINED_MODEL, MAX_LENGTH):
        super(tokenhandler, self).__init__()
        self.pre_trained_model = PRE_TRAINED_MODEL
        self.max_length = MAX_LENGTH

        self.tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL, do_lower_case=True)  # BertTokenizer, DistilBertTokenizer

    def tokenize_dataset(self, sentences):
        input_ids = []
        token_type_ids = []
        attention_masks = []

        for sent in sentences:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Instancia a codificar
                add_special_tokens=True,  # Añadir '[CLS]' y '[SEP]'
                max_length=self.max_length,  # Truncar todas las instancias
                truncation=True,
                padding='max_length',  # Añadir padding
                return_attention_mask=True,  # Construir las máscaras de atención.
                return_tensors='pt')
            
            # Add the encoded instance to the list
            input_ids.append(encoded_dict['input_ids'])
            if self.pre_trained_model == "bert-base-uncased" or self.pre_trained_model == "nghuyong/ernie-2.0-base-en":
                token_type_ids.append(encoded_dict["token_type_ids"])
            # Add the attention mask
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        if self.pre_trained_model == "bert-base-uncased" or self.pre_trained_model == "nghuyong/ernie-2.0-base-en":
            token_type_ids = torch.cat(token_type_ids, dim=0)
        
        if self.pre_trained_model == "bert-base-uncased" or self.pre_trained_model == "nghuyong/ernie-2.0-base-en":
            return input_ids, token_type_ids, attention_masks
        elif self.pre_trained_model == "distilbert-base-uncased" or self.pre_trained_model == "roberta-base":
            return input_ids, attention_masks
        
    def tokenize_dataset_two_sentences(self, sentences1, sentences2):
        input_ids = []
        token_type_ids = []
        attention_masks = []

        for i in range(len(sentences1)):
            encoded_dict = self.tokenizer.encode_plus(
                sentences1[i],  # Instancia a codificar
                sentences2[i],
                add_special_tokens=True,  # Añadir '[CLS]' y '[SEP]'
                max_length=self.max_length,  # Truncar todas las instancias
                truncation=True,
                pad_to_max_length=True,
                padding='max_length',  # Añadir padding
                return_attention_mask=True,  # Construir las máscaras de atención.
                return_tensors='pt')
            
            # Add the encoded instance to the list
            input_ids.append(encoded_dict['input_ids'])
            if self.pre_trained_model == "bert-base-uncased" or self.pre_trained_model == "nghuyong/ernie-2.0-base-en":
                token_type_ids.append(encoded_dict["token_type_ids"])
            # Add the attention mask
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        if self.pre_trained_model == "bert-base-uncased" or self.pre_trained_model == "nghuyong/ernie-2.0-base-en":
            token_type_ids = torch.cat(token_type_ids, dim=0)
        
        if self.pre_trained_model == "bert-base-uncased" or self.pre_trained_model == "nghuyong/ernie-2.0-base-en":
            return input_ids, token_type_ids, attention_masks
        elif self.pre_trained_model == "distilbert-base-uncased" or self.pre_trained_model == "roberta-base":
            return input_ids, attention_masks

    def tensor_the_dataset_bert(self, input_ids, input_token_ids, attention_masks, labels, labels2):
        new_labels = torch.tensor(labels)
        if labels2 is not None:
            new_labels2 = torch.tensor(labels2)
            tensor_dataset = TensorDataset(input_ids, input_token_ids, attention_masks, new_labels, new_labels2)
        else:
            tensor_dataset = TensorDataset(input_ids, input_token_ids, attention_masks, new_labels)
        return tensor_dataset
    
    def tensor_the_dataset_distil_bert(self, input_ids, attention_masks, labels, labels2):
        new_labels = torch.tensor(labels)
        if labels2 is not None:
            new_labels2 = torch.tensor(labels2)
            tensor_dataset = TensorDataset(input_ids, attention_masks, new_labels, new_labels2)
        else:
            tensor_dataset = TensorDataset(input_ids, attention_masks, new_labels)
        return tensor_dataset
    
    def create_dataloader_random(self, dataset, BATCH_SIZE):
        ds_dataloader = DataLoader(
            dataset,  # training instances
            sampler = RandomSampler(dataset), # Pull out batches randomly
            batch_size = BATCH_SIZE # train with this batch size.
        )
        return ds_dataloader

    def create_dataloader_weighted(self, dataset, BATCH_SIZE, weights):
        ds_dataloader = DataLoader(
            dataset,  # training instances
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True),# Pull out batches
            batch_size = BATCH_SIZE # train with this batch size.
        )
        return ds_dataloader
    
    def create_dataloader_sequential(self, dataset, BATCH_SIZE):
        ds_dataloader = DataLoader(
            dataset,  # training instances
            sampler = SequentialSampler(dataset), # Pull out batches randomly
            batch_size = BATCH_SIZE # train with this batch size.
        )
        return ds_dataloader