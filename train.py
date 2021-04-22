import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from transformers import get_cosine_schedule_with_warmup

from BERT import BERTForClassification
from ClassificationDataset import ClassificationDataset

import argparse

parser = argparse.ArgumentParser(description='fine-tune pretrained BERT model')

parser.add_argument('--num_worker', default=1, type=int,
                    help='Number of workers for dataloader')

parser.add_argument('-f', '--model_folder', default="models", type=str,
                    help='Folder to save models')
parser.add_argument('-n', '--model_name', default="pretrain", type=str,
                    help='Prefix of model name')
parser.add_argument('-p', '--pretrain_state_dict_file', default=None, type=str,
                    help='Pretrain model state dict')
parser.add_argument('-s', '--start_epoch', default=1, type=int,
                    help='Start epoch')
parser.add_argument('-e', '--num_epoch', default=5, type=int,
                    help='Number of epoch used to train')


parser.add_argument('-d', '--dataset_name', default="yelp", type=str,
                    help='Dataset used to fine-tune')
parser.add_argument('-c', '--num_class', default=2, type=int,
                    help='Number of classes in the dataset')
parser.add_argument('-l', '--sent_length', default=510, type=int,
                    help='Sent length used to pretrain')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='Batch size used to pretrain')
parser.add_argument('-r', '--learning_rate', default=2e-3, type=float,
                    help='Learning rate used to pretrain')
parser.add_argument('-w', '--warmup_proportion', default=0.1, type=float,
                    help='Warmup Proportion used to pretrain')


args = parser.parse_args()


def train(data_loader, model, criterion, optimizer, scheduler, device, print_every=100):
    model.train()
    total_loss = 0.0

    for step, (seq, attn_masks, labels) in enumerate(data_loader):
        seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(seq, attn_masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if (step + 1) % print_every == 0:
            print('Step [%d / %d] loss: %f' % (step + 1, len(data_loader), loss.item()), flush=True)
    return total_loss


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


def save_model(model, model_folder, prefix, suffix):
    """
    Save model state dict to file

    :param model: model to be saved
    :param model_folder: folder to save model
    :param prefix: prefix of model name
    :param suffix: suffix of model name
    :return: None
    """

    import os
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    out = os.path.join(model_folder, "{}_checkpoint_{}.tar".format(prefix, suffix))

    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model_folder = args.model_folder
    model_name = args.model_name
    pretrain_state_dict_file = args.pretrain_state_dict_file
    start_epoch = args.start_epoch
    num_epoch = args.num_epoch

    # Model parameter
    dataset_name = args.dataset_name
    num_class = args.num_class
    sent_length = args.sent_length
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    warmup_proportion = args.warmup_proportion

    train_dataset = ClassificationDataset(dataset_name, 'train', sent_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=num_worker, pin_memory=True)

    test_dataset = ClassificationDataset(dataset_name, 'test', sent_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False, num_workers=num_worker, pin_memory=True)

    model = BERTForClassification(num_class)
    if pretrain_state_dict_file:  # load pretrain bert data
        file_path=f"{model_folder}/{pretrain_state_dict_file}"
        state_dict = torch.load(file_path, map_location=device)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        model.load_state_dict(state_dict, strict=False)
    elif start_epoch != 1:  # load previous train data
        file_path = f"{model_folder}/{model_name}-{dataset_name}-{batch_size}-{learning_rate}-{warmup_proportion}_checkpoint_{start_epoch - 1}.tar"
        model.load_state_dict(torch.load(file_path, map_location=device))
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    num_steps_per_epoch = len(train_loader)
    num_steps = num_steps_per_epoch * num_epoch
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_proportion * num_steps, num_steps)

    for epoch in range(start_epoch, num_epoch + 1):
        loss_epoch = train(train_loader, model, criterion, optimizer, lr_scheduler, device)
        print(f"Epoch [{epoch} / {num_epoch}]\t Loss: {loss_epoch / len(train_loader)}", flush=True)

        # save and test model
        save_model(model, model_folder, f"{model_name}-{dataset_name}-{batch_size}-{learning_rate}-{warmup_proportion}",
                   epoch)
        accuracy = test(test_loader, model, device)
        print(f"Epoch {epoch} model saved, accuracy: {accuracy}", flush=True)
