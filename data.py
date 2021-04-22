from transformers import AutoTokenizer

from torchtext.datasets import YahooAnswers, YelpReviewPolarity, DBpedia, IMDB, AmazonReviewFull, AG_NEWS

import os, pickle

dataset_name, split = "yahoo", "train"

if dataset_name == "yahoo":
    data_iter = YahooAnswers(split=split)
elif dataset_name == "yelp":
    data_iter = YelpReviewPolarity(split=split)
elif dataset_name == "dbpedia":
    data_iter = DBpedia(split=split)
elif dataset_name == "imdb":
    data_iter = IMDB(split=split)
elif dataset_name == "amazon":
    data_iter = AmazonReviewFull(split=split)
elif dataset_name == "agnews":
    data_iter = AG_NEWS(split=split)
else:
    raise NotImplementedError("No such dataset")

tokens, labels = [], []

for label, line in data_iter:
    tokens.append(line)
    if dataset_name != "imdb":
        labels.append(label - 1)
    else:
        labels.append(1 if label == "pos" else 0)

# Encode sentences
tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
outputs = tok.batch_encode_plus(tokens)

res = {'labels': labels, "tokens_ids": outputs["input_ids"]}

if not os.path.exists('datasets/'):
    os.mkdir('datasets/')
with open(f'datasets/{dataset_name}_{split}.pkl', 'wb') as f:
    pickle.dump(res, f)
