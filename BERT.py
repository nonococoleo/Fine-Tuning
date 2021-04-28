import torch.nn as nn
import torch.nn.functional as F

from transformers import DistilBertModel


class BERTForClassification(nn.Module):
    """
    DistilBERT model with classification header
    """

    def __init__(self, num_classes, freeze_bert_paras=True, model_name='distilbert-base-uncased'):
        """
        Initialize model

        :param num_classes: number of classes in classification task
        :param freeze_bert_paras: whether free BERT parameters
        :param model_name: base DistilBERT model
        """

        super(BERTForClassification, self).__init__()

        # Load base DistilBERT model
        self.bert_layer = DistilBertModel.from_pretrained(model_name)

        # Freeze bert layers
        if freeze_bert_paras:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.fc = nn.Linear(768, num_classes)

    def forward(self, sequence, attention_masks):
        """
        Forward Propagation

        :param sequence: sentences
        :param attention_masks: attention masks
        :return: output
        """

        # Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(sequence, attention_mask=attention_masks)

        # Use hidden states of [CLS] for classification
        out = outputs.last_hidden_state[:, 0, :]

        # Dropout
        out = F.dropout(out, p=0.1)

        # Feeding cls_rep to the classifier layer
        out = self.fc(out)

        return out
