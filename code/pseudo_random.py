import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from tqdm import tqdm

from pseudo_label_method import pseudo_label_random
from utils import Dataset
from create_bags import create_bags
from load_mnist import load_minist
from load_cifar10 import load_cifar10

log = logging.getLogger(__name__)


def train(model, criterion, optimizer, loader, device):
    model.train()
    loss_list, acc_list = [], []
    for data, label in loader:
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


def evaluation(model, criterion, loader, device):
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


@ hydra.main(config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:

    log.info(OmegaConf.to_yaml(cfg))
    cwd = hydra.utils.get_original_cwd()
    log.info('cwd:%s' % cwd)

    num_bags = cfg.dataset.num_bags
    num_instances = cfg.dataset.num_instances

    # load data
    if cfg.dataset.name == 'MNIST':
        train_data, train_label, test_data, test_label = \
            load_minist(
                dataset_dir=cwd+cfg.dataset.dir,
                is_to_rgb=cfg.dataset.is_to_rgb)
    if cfg.dataset.name == 'CIFAR10':
        train_data, train_label, test_data, test_label = \
            load_cifar10(
                dataset_dir=cwd+cfg.dataset.dir
            )

    negative_data = test_data[test_label == cfg.dataset.negative_label]
    positive_data = test_data[test_label == cfg.dataset.positive_label]
    num_negative = sum(test_label == cfg.dataset.negative_label)
    num_positive = sum(test_label == cfg.dataset.positive_label)
    test_data = np.concatenate([negative_data, positive_data])
    test_label = np.concatenate(
        [np.zeros(num_negative), np.ones(num_positive)])
    test_dataset = Dataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)

    # create bugs
    bags_data, bags_label, label_proportion = create_bags(
        train_data, train_label,
        num_bags=num_bags,
        num_instances=num_instances,
        negative_label=cfg.dataset.negative_label,
        positive_label=cfg.dataset.positive_label)

    N = cfg.dataset.img_size
    train_data = bags_data.reshape(-1, N, N, cfg.dataset.img_ch)
    train_label = bags_label.reshape(-1)
    print(train_data.shape, train_label.shape)

    # define model
    model = resnet18(pretrained=cfg.is_pretrained)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(cfg.device)

    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    test_loss_list, test_acc_list = [], []
    for epoch in tqdm(range(cfg.num_epochs)):
        # update pseudo label
        pseudo_label = pseudo_label_random(
            label_proportion, num_bags, num_instances)

        # trained DNN: obtain f_t with d_t
        train_dataset = Dataset(train_data, pseudo_label)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            shuffle=True,  num_workers=cfg.num_workers)

        train_loss, train_acc = train(model=model,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      loader=train_loader,
                                      device=cfg.device)
        # print(train_loss, train_acc)

        test_loss, test_acc = evaluation(model=model,
                                         criterion=criterion,
                                         loader=test_loader,
                                         device=cfg.device)
        print('test_loss: %.4f, test_acc: %.4f' % (test_loss, test_acc))
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

    np.save('%s/result/loss_random_%s_%s_%s_%s_%s' %
            (cwd, cfg.dataset.name,
             cfg.dataset.negative_label,
             cfg.dataset.positive_label,
             cfg.dataset.num_bags,
             cfg.dataset.num_instances),
            np.array(test_loss_list))

    np.save('%s/result/acc_random_%s_%s_%s_%s_%s' %
            (cwd, cfg.dataset.name,
             cfg.dataset.negative_label,
             cfg.dataset.positive_label,
             cfg.dataset.num_bags,
             cfg.dataset.num_instances),
            np.array(test_acc_list))


if __name__ == '__main__':
    main()
