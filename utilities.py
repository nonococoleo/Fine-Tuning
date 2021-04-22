import os

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
