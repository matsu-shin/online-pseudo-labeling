import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as transforms


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = self.label.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]

        data = self.transform(data)
        label = torch.tensor(label).long()

        return data, label


def evaluate(model, criterion, loader, device):
    model.eval()
    loss_list, acc_list = [], []
    pred_list, feature_list = [], []
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        model.avgpool.register_forward_hook(store_feature)
        y = model(data)
        feature_list.extend(feature.squeeze().cpu().detach().numpy())
        loss = criterion(y, label)

        loss_list.append(loss.item())
        y, label = y.cpu().detach().numpy(), label.cpu().detach().numpy()
        pred_list.extend(y.argmax(1))
        acc_list.extend(y.argmax(1) == label)

    loss = np.array(loss_list).mean()
    acc = np.array(acc_list).mean()
    pred = np.array(pred_list)
    features = np.array(feature_list)

    return loss, acc, pred, features


def store_feature(module, input, output):
    global feature
    feature = output
