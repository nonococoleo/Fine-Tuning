import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from transformers import get_cosine_schedule_with_warmup

from BERT import BERTForClassification
from ClassificationDataset import ClassificationDataset

import argparse

parser = argparse.ArgumentParser(description='Further pretrain BERT model')

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


args = parser.parse_args()


def test(data_loader, model, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for seq, attn_masks, labels in data_loader:
            seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)
            outputs = model(seq, attn_masks)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def save_model(model, model_path, prefix, num_steps):
    import os
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    out = os.path.join(model_path, "{}_{}.tar".format(prefix, num_steps))

    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model_folder = args.model_folder
    model_name = args.model_name
    num_steps = args.num_steps

    # Model parameter
    dataset_name = args.dataset_name
    num_class = args.num_class
    sent_length = 510
    batch_size = 64
    learning_rate = 2e-5
    warmup_proportion = 0.1

    train_dataset = ClassificationDataset(dataset_name, 'train', sent_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    test_dataset = ClassificationDataset(dataset_name, 'test', sent_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

    model = BERTForClassification(num_class, freeze=False)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_proportion * num_steps, num_steps)

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

    # save and test model
    save_model(model, model_folder, f"{model_name}-{dataset_name}-{batch_size}-{learning_rate}", num_steps)
    accuracy = test(test_loader, model, device)
    print(f"Step {num_steps} pretrain model saved, accuracy: {accuracy}", flush=True)
