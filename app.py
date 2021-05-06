from transformers import AutoTokenizer

from utilities import *
from BERT import BERTForClassification

import argparse

parser = argparse.ArgumentParser(description='RESTful Sentiment Analysis API')

parser.add_argument('-m', '--model_path', default="models/all_32-0.002-0.1_checkpoint_5.tar", type=str,
                    help='Path to Sentiment Analysis model')

parser.add_argument('-d', '--host', default='0.0.0.0', type=str,
                    help='Host of server')

parser.add_argument('-p', '--port', default=5000, type=int,
                    help='Port of server')

args = parser.parse_args()

model_file = args.model_path


def load_model(device, model_file, num_classes=2):
    model = BERTForClassification(num_classes)
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    return model


def encode_sentences(sentences, tokenizer, max_sequence_length=512, max_sentence_length=510):
    sentences = tokenizer.batch_encode_plus(sentences)["input_ids"]

    tokens_ids, attention_masks = [], []
    for sentence in sentences:
        tokens_id, attention_mask = process_inputs(sentence, max_sentence_length, max_sequence_length)

        tokens_ids.append(tokens_id)
        attention_masks.append(attention_mask)

    return torch.Tensor(tokens_ids).long(), torch.Tensor(attention_masks).int()


def predict(device, model, seq, attn_masks):
    with torch.no_grad():
        seq, attn_masks = seq.to(device), attn_masks.to(device)
        outputs = model(seq, attn_masks)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.bool().tolist()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = load_model(device, model_file)
print("model loaded")

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
print("tokenizer loaded")


def get_results(sentences, tok=tokenizer, model=model):
    tokens_ids, attention_masks = encode_sentences(sentences, tok)
    return predict(device, model, tokens_ids, attention_masks)


import logging
from flask import Flask
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
cors = CORS(app)
api = Api(app)


class HealthCheck(Resource):
    def get(self):
        return {'status': 'alive'}


parser = reqparse.RequestParser()
parser.add_argument('sentences', type=list, location='json', required=True,
                    help="Input sentences to be predicted")


class Predict(Resource):
    def post(self):
        args = parser.parse_args()
        inputs = args['sentences']
        results = get_results(inputs)
        message = {"sentences": inputs, "labels": results}
        app.logger.info(f'api predict: {message}')
        return {"success": True, "message": message}


api.add_resource(HealthCheck, '/')
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.logger.setLevel(logging.INFO)
    app.run(args.host, args.port)
