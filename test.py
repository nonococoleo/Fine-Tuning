from torch.utils.data import DataLoader

from utilities import *
from BERT import BERTForClassification
from ClassificationDataset import ClassificationDataset

import argparse

parser = argparse.ArgumentParser(description='Test BERT model')

parser.add_argument('--num_workers', default=1, type=int,
                    help='Number of workers for dataloader')
parser.add_argument('-f', '--model_folder', default="models", type=str,
                    help='Folder to save models')
parser.add_argument('-t', '--state_dict_file', default=None, type=str,
                    help='Pre-trained model state used to test')

parser.add_argument('-d', '--dataset_name', default="imdb", type=str,
                    help='Dataset used to test')
parser.add_argument('-c', '--num_class', default=2, type=int,
                    help='Number of classes in the dataset')
parser.add_argument('-l', '--sent_length', default=510, type=int,
                    help='Sent length used to test')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='Batch size used to test')

args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Model settings
    num_workers = args.num_workers
    model_folder = args.model_folder
    state_dict_file = args.state_dict_file

    # Test parameter
    dataset_name = args.dataset_name
    num_classes = args.num_class
    sent_length = args.sent_length
    batch_size = args.batch_size

    test_dataset = ClassificationDataset(dataset_name, 'test', sent_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    model = BERTForClassification(num_classes)
    file_path = f"{model_folder}/{state_dict_file}"
    state_dict = torch.load(file_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    accuracy = test(test_loader, model, device)
    print(f"accuracy: {accuracy}", flush=True)
