import os
import pickle

import torch


def train(data_loader, model, criterion, optimizer, scheduler, device, print_every=100):
    """
    Train model

    :param data_loader: dataloader of target dataset
    :param model: model to be trained
    :param criterion: Loss function
    :param optimizer: objective function
    :param scheduler: learning rate scheduler
    :param device: model running device
    :param print_every: print interval
    :return: total training loss
    """

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
    """
    Test model accuracy

    :param data_loader: dataloader of target dataset
    :param model: model to be tested
    :param device: model running device
    :return: model accuracy on target dataset
    """

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

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    out = os.path.join(model_folder, "{}_checkpoint_{}.tar".format(prefix, suffix))

    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)


def process_inputs(tokens_id, max_sentence_length=510, max_sequence_length=512):
    """
    Truncate long sentences and pad to max_sequence_length
    :param tokens_id: encoded sentence
    :param max_sentence_length: max sentence length
    :param max_sequence_length: max sequence length
    :return: sequence and attention mask input for model
    """

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

    return tokens_id, attention_mask


def get_dataset(dataset_name, split):
    """
    Load encoded dataset by name
    :param dataset_name: target dataset name
    :param split: dataset split
    :return: list of data
    """

    if dataset_name.lower() in ["imdb", "yelp", "yahoo", "agnews", "amazon", "dbpedia"] \
            and split in ["train", "test"]:
        with open(f"datasets/{dataset_name.lower()}_{split}.pkl", "rb") as f:
            data = pickle.load(f)
            return data
    else:
        raise NotImplementedError("No Such dataset")
