import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from tqdm import tqdm

import torchvision.transforms as transforms
from evaluate import evaluate
from utils import Dataset
from create_bags import create_bags, get_label_proportion
from load_mnist import load_minist
from load_cifar10 import load_cifar10

log = logging.getLogger(__name__)


class Bag_Dataset(torch.utils.data.Dataset):
    def __init__(self, bags, label_proportions):
        self.bags = bags
        self.label_proportions = label_proportions
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.len = self.bags.shape[0]
        self.num_instances = self.bags.shape[1]
        self.img_size = self.bags.shape[2]
        self.img_ch = self.bags.shape[-1]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        bag = torch.zeros((self.num_instances, self.img_ch,
                          self.img_size, self.img_size))
        for i in range(self.num_instances):
            bag[i] = self.transform(self.bags[idx][i])
        shuffle_index = np.arange(self.num_instances)
        np.random.shuffle(shuffle_index)
        bag = bag[shuffle_index]

        label_proportion = self.label_proportions[idx]
        label_proportion = torch.tensor(label_proportion).float()

        return bag, label_proportion


def train(model, optimizer, loader, device):
    model.train()
    loss_list, acc_list = [], []
    print('training model...')
    loss_function = nn.MSELoss()
    for bag, lp in tqdm(loader):
        bag, lp = bag[0].to(device), lp[0].to(device)
        y = model(bag)
        num_instances = bag.size(0)
        estimated_lp = 1/num_instances * F.softmax(y, dim=1).sum(0)
        # loss = - torch.sum(lp * torch.log(estimated_lp))
        loss = F.l1_loss(lp, estimated_lp, reduction="none").mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_list.append(loss.item())

    loss = np.array(loss_list).mean()

    return loss


@ hydra.main(config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:

    log.info(OmegaConf.to_yaml(cfg))
    cwd = hydra.utils.get_original_cwd()
    log.info('cwd:%s' % cwd)

    num_bags = cfg.dataset.num_bags
    num_instances = cfg.dataset.num_instances
    num_classes = cfg.dataset.num_classes

    # load data
    if cfg.dataset.name == 'mnist':
        train_data, train_label, test_data, test_label = \
            load_minist(
                dataset_dir=cwd+cfg.dataset.dir,
                is_to_rgb=cfg.dataset.is_to_rgb)
    elif cfg.dataset.name == 'cifar10':
        train_data, train_label, test_data, test_label = \
            load_cifar10(
                dataset_dir=cwd+cfg.dataset.dir
            )

    if cfg.is_load_bag:
        bags_index = np.load(cwd+'/obj/%s/uniform-SWOR-%s.npy' %
                             (cfg.dataset.name, cfg.dataset.num_instances))
        cfg.dataset.num_classes = 10
        cfg.dataset.num_bags = bags_index.shape[0]
        num_classes = cfg.dataset.num_classes
        num_instances = cfg.dataset.num_instances
        num_bags = cfg.dataset.num_bags

        bags_data, bags_label = train_data[bags_index], train_label[bags_index]
        label_proportion = np.zeros((num_bags, num_classes))
        for n in range(cfg.dataset.num_bags):
            bag_one_hot_label = np.identity(num_classes)[bags_label[n]]
            label_proportion[n] = bag_one_hot_label.sum(axis=0)/num_instances
    else:
        num_classes = cfg.dataset.num_classes
        num_instances = cfg.dataset.num_instances
        num_bags = cfg.dataset.num_bags
        # create bugs
        label_proportion = get_label_proportion(
            num_bags=num_bags, num_instances=num_instances, num_classes=num_classes)
        # label_proportion.shape => (num_bags, num_classes)

        bags_data, bags_label = create_bags(
            train_data, train_label, label_proportion,
            num_bags=num_bags,
            num_instances=num_instances)
        # bags_data.shape => (num_bags, num_instances, (image_size))
        # bags_label.shape => (num_bags, num_instances)

    N = cfg.dataset.img_size
    train_data = bags_data.reshape(-1, N, N, cfg.dataset.img_ch)
    train_label = bags_label.reshape(-1)
    # train_data.shape => (num_bags*num_instances, (image_size))
    # train_label.shape => (num_bags*num_instances)

    # for test
    data, label = [], []
    for c in range(num_classes):
        data.extend(test_data[test_label == c])
        label.extend(test_label[test_label == c])
    test_data, test_label = np.array(data), np.array(label)
    print(test_data.shape, test_label.shape)
    test_dataset = Dataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)

    train_dataset = Bag_Dataset(bags_data, label_proportion)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1,
        shuffle=True,  num_workers=cfg.num_workers)

    # define model
    model = resnet18(pretrained=cfg.is_pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(cfg.device)

    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    test_loss_list, test_acc_list = [], []
    for epoch in range(cfg.num_epochs):
        # TODO: need to implement mini batch
        train(model=model,
              optimizer=optimizer,
              loader=train_loader,
              device=cfg.device)

        test_loss, test_acc = evaluate(model=model,
                                       criterion=criterion,
                                       loader=test_loader,
                                       device=cfg.device)

        # print result
        print('[Epoch: %d/%d] test_loss: %.4f, test_acc: %.4f' %
              (epoch+1, cfg.num_epochs, test_loss, test_acc))
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

    # save
    f_name = 'proportion_loss'
    f_name += '_%s_%s_%s_%s' % (
        cfg.dataset.name,
        cfg.dataset.num_classes,
        cfg.dataset.num_bags,
        cfg.dataset.num_instances)
    np.save('% s/result/loss_%s' % (cwd, f_name), np.array(test_loss_list))
    np.save('% s/result/acc_%s' % (cwd, f_name), np.array(test_acc_list))


if __name__ == '__main__':
    main()
