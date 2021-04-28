from torch.utils.data import Dataset

from transformers import AutoTokenizer

from utilities import *


class ClassificationDataset(Dataset):
    """
    Uniformed dataset for classification
    """

    def __init__(self, dataset_name, split, max_sentence_length, model_name='distilbert-base-uncased',
                 max_sequence_length=512):
        """
        Initialize dataset

        :param dataset_name: target dataset name
        :param split: dataset split
        :param max_sentence_length: max sentence length used
        :param model_name: target model name
        :param max_sequence_length: max sequence length allowed in model
        """

        # Store the contents of the file in a pandas dataframe
        data = get_dataset(dataset_name, split)
        self.labels = data["labels"]
        self.tokens_ids = data["tokens_ids"]

        # Initialize the BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_sequence_length = max_sequence_length

        self.max_sentence_length = max_sentence_length
        self.label_ids = {}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        tokens_id = self.tokens_ids[index]

        tokens_id_tensor, attention_mask = process_inputs(tokens_id, self.max_sentence_length, self.max_sequence_length)

        return tokens_id_tensor, attention_mask, label
