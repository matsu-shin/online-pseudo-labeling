import numpy as np
from tqdm import tqdm


def evaluate(model, criterion, loader, device):
    model.eval()
    loss_list, acc_list = [], []
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        y = model(data)
        loss = criterion(y, label)

        loss_list.append(loss.item())
        y, label = y.cpu().detach().numpy(), label.cpu().detach().numpy()
        acc_list.extend(y.argmax(1) == label)

    loss = np.array(loss_list).mean()
    acc = np.array(acc_list).mean()

    return loss, acc
