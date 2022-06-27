from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
import glob
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.len = self.label.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]

        data = self.transform(data)
        label = torch.tensor(label).long()

        return data, label


def show_figure():
    cwd = './result/'

    for path in glob.glob(cwd+'loss_*'):
        plt.plot(np.load(path), label=path)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(cwd+'loss.png')
    plt.close()

    for path in glob.glob(cwd+'acc_*'):
        plt.plot(np.load(path), label=path)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig(cwd+'acc.png')
    plt.close()


if __name__ == '__main__':
    show_figure()
