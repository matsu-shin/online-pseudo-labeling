import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, pseudo_label, proportion):
        self.data = data
        self.label = label
        self.pseudo_label = pseudo_label
        self.proportion = proportion
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        pseudo_label = self.pseudo_label[idx]
        proportion = self.proportion[idx]

        (n, w, h, c) = data.shape
        normalized_data = torch.zeros((n, c, w, h))
        for i in range(data.shape[0]):
            normalized_data[i] = self.transform(data[i])
        normalized_data = normalized_data.float()
        label = torch.tensor(label).long()
        pseudo_label = torch.tensor(pseudo_label).long()
        proportion = torch.tensor(proportion).float()

        return normalized_data, label, pseudo_label, proportion


def train(model, ce_loss_f, proportion_loss_f, optimizer, loader, cfg):
    model.train()
    ce_losses, proportion_losses = [], []
    print('training model...')
    for data, _, pseudo_label, proportion in tqdm(loader):
        data, pseudo_label = data.to(cfg.device), pseudo_label.to(cfg.device)
        proportion = proportion.to(cfg.device)

        (n, b, w, h, c) = data.size()
        data = data.reshape(-1, w, h, c)
        pseudo_label = pseudo_label.reshape(-1)

        y = model(data)
        confidence = F.softmax(y, dim=1)
        confidence = confidence.reshape(n, b, -1).mean(dim=1)

        ce_loss = cfg.w_ce*ce_loss_f(y, pseudo_label)
        proportion_loss = cfg.w_proportion * \
            proportion_loss_f(confidence, proportion)
        loss = ce_loss + proportion_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        ce_losses.append(ce_loss.item())
        proportion_losses.append(proportion_loss.item())

    return np.array(ce_losses).mean(), np.array(proportion_losses).mean()
