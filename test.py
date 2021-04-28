from torch.utils.data import DataLoader

from utilities import *
from BERT import BERTForClassification
from ClassificationDataset import ClassificationDataset

import argparse

parser = argparse.ArgumentParser(description='Test BERT model')

parser.add_argument('--num_workers', default=1, type=int,
                    help='Number of workers for dataloader')

args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Model settings
    num_workers = args.num_workers
    model_folder = "models"
    state_dict_file = "pretrain/pretrain-yelp-64-0.00002_10000.tar"

    # Test parameter
    dataset_name = "imdb"
    num_classes = 2
    sent_length = 510
    batch_size = 32

    test_dataset = ClassificationDataset(dataset_name, 'test', sent_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

    model = BERTForClassification(num_classes)
    file_path = f"{model_folder}/{state_dict_file}"
    state_dict = torch.load(file_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    accuracy = test(test_loader, model, device)
    print(f"accuracy: {accuracy}", flush=True)
