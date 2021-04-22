import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel


class BERTForClassification(nn.Module):
    def __init__(self, num_class, freeze=True, model_name='distilbert-base-uncased'):
        super(BERTForClassification, self).__init__()

        self.bert_layer = DistilBertModel.from_pretrained(model_name)

        # Freeze bert layers
        if freeze:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.fc = nn.Linear(768, num_class)

    def forward(self, sequence, attention_masks):
        # Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(sequence, attention_mask=attention_masks)

        # Use hidden states of [CLS] for classification
        out = outputs.last_hidden_state[:, 0, :]

        out = F.dropout(out, p=0.1)

        # Feeding cls_rep to the classifier layer
        out = self.fc(out)

        return out
