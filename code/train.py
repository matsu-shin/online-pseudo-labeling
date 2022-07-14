import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as transforms


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, pseudo_label):
        self.data = data
        self.label = label
        self.pseudo_label = pseudo_label
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = self.label.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        pseudo_label = self.pseudo_label[idx]

        data = self.transform(data)
        label = torch.tensor(label).long()
        pseudo_label = torch.tensor(pseudo_label).long()

        return data, label, pseudo_label


def train(model, criterion, optimizer, loader, device):
    model.train()
    # loss_list, acc_list = [], []
    # pseudo_loss_list, pseudo_acc_list = [], []
    print('training model...')
    for data, _, pseudo_label in tqdm(loader):
        data, pseudo_label = data.to(device), pseudo_label.to(device)
        y = model(data)
        pseudo_loss = criterion(y, pseudo_label)
        pseudo_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    #     pseudo_loss_list.append(pseudo_loss.item())
    #     loss_list.append(criterion(y, label).item())

    #     y, pseudo_label = y.cpu().detach().numpy(), pseudo_label.cpu().detach().numpy()
    #     acc_list.extend(y.argmax(1) == label)
    #     pseudo_acc_list.extend(y.argmax(1) == pseudo_label)

    # loss = np.array(loss_list).mean()
    # acc = np.array(acc_list).mean()
    # pseudo_acc = np.array(pseudo_acc_list).mean()

    # return loss, acc, pseudo_acc
