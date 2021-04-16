import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

import pickle


def get_dataset(dataset_name, split):
    if dataset_name.lower() in ["imdb", "yelp", "yahoo"]:
        with open(f"datasets/{dataset_name.lower()}_{split}.pkl", "rb") as f:
            data = pickle.load(f)
            return data
    else:
        raise NotImplementedError


class ClassificationDataset(Dataset):
    def __init__(self, dataset_name, split, max_sentence_length, model_name='distilbert-base-uncased'):
        # Store the contents of the file in a pandas dataframe
        data = get_dataset(dataset_name, split)
        self.labels = data["labels"]
        self.tokens_ids = data["tokens_ids"]

        # Initialize the BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_sequence_length = 512

        self.max_sentence_length = max_sentence_length
        self.label_ids = {}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        tokens_id = self.tokens_ids[index]

        # truncate long sentences
        if len(tokens_id) < self.max_sentence_length + 1:
            tokens_id = tokens_id[:self.max_sentence_length + 1] + [102]

        if len(tokens_id) < self.max_sequence_length:
            # Padding sentences
            tokens_id = tokens_id + [0 for _ in range(self.max_sequence_length - len(tokens_id))]
        else:
            # Pruning the list to be of specified max length
            tokens_id = tokens_id[:self.max_sequence_length - 1] + [102]

        # Converting the list to a pytorch tensor
        tokens_id_tensor = torch.tensor(tokens_id)

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (tokens_id_tensor != 0).long()

        return tokens_id_tensor, attention_mask, label
