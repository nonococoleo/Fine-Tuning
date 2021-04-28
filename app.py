from utilities import *
from transformers import AutoTokenizer
from BERT import BERTForClassification

model_file = "models/all_10000-yelp-32-0.002-0.1_checkpoint_5.tar"


def load_model(device, model_file, num_classes=2):
    model = BERTForClassification(num_classes, freeze=False)
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    return model


def encode_sentences(sentences, tokenizer, max_sequence_length=512, max_sentence_length=510):
    sentences = tokenizer.batch_encode_plus(sentences)["input_ids"]

    tokens_ids, attention_masks = [], []
    for tokens_id in sentences:
        # truncate long sentences
        if len(tokens_id) < max_sentence_length + 1:
            tokens_id = tokens_id[:max_sentence_length + 1] + [102]

        if len(tokens_id) < max_sequence_length:
            # Padding sentences
            tokens_id = tokens_id + [0 for _ in range(max_sequence_length - len(tokens_id))]
        else:
            # Pruning the list to be of specified max length
            tokens_id = tokens_id[:max_sequence_length - 1] + [102]

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = [1 if x != 0 else 0 for x in tokens_id]

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
    app.run('0.0.0.0')
