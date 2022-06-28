import numpy as np
from tqdm import tqdm


def train(model, criterion, optimizer, loader, device):
    model.train()
    loss_list, acc_list = [], []
    print('training model...')
    for data, label in tqdm(loader):
        data, label = data.to(device), label.to(device)
        y = model(data)
        loss = criterion(y, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_list.append(loss.item())
        y, label = y.cpu().detach().numpy(), label.cpu().detach().numpy()
        acc_list.extend(y.argmax(1) == label)

    loss = np.array(loss_list).mean()
    acc = np.array(acc_list).mean()

    return loss, acc
