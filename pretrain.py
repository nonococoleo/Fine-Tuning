from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from transformers import get_cosine_schedule_with_warmup

from utilities import *
from BERT import BERTForClassification
from ClassificationDataset import ClassificationDataset

import argparse

parser = argparse.ArgumentParser(description='Further pretrain BERT model')

parser.add_argument('--num_worker', default=1, type=int,
                    help='Number of workers for dataloader')

parser.add_argument('-f', '--model_folder', default="models", type=str,
                    help='Folder to save models')
parser.add_argument('-n', '--model_name', default="pretrain", type=str,
                    help='Prefix of model name')
parser.add_argument('-s', '--num_steps', default=10000, type=int,
                    help='Number of further pretrain steps')

parser.add_argument('-d', '--dataset_name', default="yelp", type=str,
                    help='Dataset used to pretrain')
parser.add_argument('-c', '--num_class', default=2, type=int,
                    help='Number of classes in the dataset')
parser.add_argument('-l', '--sent_length', default=510, type=int,
                    help='Sent length used to pretrain')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    help='Batch size used to pretrain')
parser.add_argument('-r', '--learning_rate', default=2e-5, type=float,
                    help='Learning rate used to pretrain')
parser.add_argument('-w', '--warmup_proportion', default=0.1, type=float,
                    help='Warmup Proportion used to pretrain')

args = parser.parse_args()

if __name__ == '__main__':
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Model settings
    num_worker = args.num_worker
    model_folder = args.model_folder
    model_name = args.model_name

    # Training parameter
    num_steps = args.num_steps
    dataset_name = args.dataset_name
    num_class = args.num_class
    sent_length = args.sent_length
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    warmup_proportion = args.warmup_proportion

    # Load target dataset
    train_dataset = ClassificationDataset(dataset_name, 'train', sent_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=num_worker, pin_memory=True)

    test_dataset = ClassificationDataset(dataset_name, 'test', sent_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False, num_workers=num_worker, pin_memory=True)

    # Load DistilBERT base model (uncased)
    model = BERTForClassification(num_class, freeze=False)
    if num_worker > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_proportion * num_steps, num_steps)

    # Further pretrain model
    model.train()
    count_steps = 0
    while count_steps < num_steps:
        for step, (seq, attn_masks, labels) in enumerate(train_loader):
            seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(seq, attn_masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            count_steps += 1

            if count_steps % 100 == 0:
                print(f"Step [{count_steps} / {num_steps}] loss: {loss}", flush=True)

            if count_steps == num_steps:
                break

    # Save and test model
    save_model(model, model_folder, f"{model_name}-{dataset_name}-{batch_size}-{learning_rate}", num_steps)
    accuracy = test(test_loader, model, device)
    print(f"{model_name}-{dataset_name}-{num_steps} model saved, accuracy: {accuracy}")
