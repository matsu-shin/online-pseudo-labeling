from math import nan
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
import torchvision.transforms as transforms

from FPL import FPL
from utils import Dataset, fix_seed, make_folder
from load_mnist import load_minist
from load_cifar10 import load_cifar10

log = logging.getLogger(__name__)


class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, data, label, p_label):
        self.data = data
        self.label = label
        self.p_label = p_label
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.transform(self.data[idx])
        label = torch.tensor(self.label[idx]).long()
        p_label = torch.tensor(self.p_label[idx]).long()
        return data, label, p_label


@ hydra.main(config_path='../config', config_name='config_toy_fpl')
def main(cfg: DictConfig) -> None:
    fix_seed(cfg.seed)

    # file name
    cwd = hydra.utils.get_original_cwd()
    result_path = cwd + cfg.result_path
    result_path += 'toy_fpl/'
    make_folder(result_path)
    result_path += '%s' % (cfg.dataset.name)
    result_path += 'p_ratio_%s' % (cfg.pseudo_ratio)
    result_path += '/'
    make_folder(result_path)

    fh = logging.FileHandler(result_path+'exec.log')
    log.addHandler(fh)
    log.info(OmegaConf.to_yaml(cfg))
    log.info('cwd:%s' % cwd)

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

    dataset_path = cwd + '/obj/%s/bias-1024-index.npy' % (cfg.dataset.name)
    print('loading...  '+dataset_path)
    bags_index = np.load(dataset_path)
    num_classes = cfg.dataset.num_classes
    num_bags = bags_index.shape[0]
    num_instances = bags_index.shape[1]

    bags_data, bags_label = train_data[bags_index], train_label[bags_index]
    label_proportion = np.zeros((num_bags, num_classes))
    for n in range(num_bags):
        bag_one_hot_label = np.identity(num_classes)[bags_label[n]]
        label_proportion[n] = bag_one_hot_label.sum(axis=0)/num_instances

    test_dataset = Dataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)

    # define model
    fix_seed(cfg.seed)
    model = resnet18(pretrained=cfg.is_pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(cfg.device)

    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    fpl = FPL(label_proportion=label_proportion,
              num_bags=num_bags, num_instances=num_instances,
              sigma=cfg.fpl.sigma,
              eta=cfg.fpl.eta,
              loss_f=cfg.fpl.loss_f,
              pseudo_ratio=cfg.pseudo_ratio,
              is_online_prediction=cfg.fpl.is_online_prediction)

    (n, b, w, h, c) = bags_data.shape
    train_data = bags_data.reshape(-1, w, h, c)
    train_label = bags_label.reshape(-1)
    train_dataset_for_fpl = Dataset(train_data, train_label)
    train_loader_for_fpl = torch.utils.data.DataLoader(
        train_dataset_for_fpl, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)

    train_loss_list, train_acc_list = [], []
    train_pseudo_acc_list = []
    pseudo_label_acc_list = []
    test_loss_list, test_acc_list = [], []
    flip_p_label_ratioes = []

    fix_seed(cfg.seed)
    for epoch in range(cfg.num_epochs):
        s_time = time.time()

        # update the noise-risk vector theta_t
        fpl.update_theta(model=model,
                         criterion=criterion,
                         loader=train_loader_for_fpl,
                         device=cfg.device)

        # select the next k-set d_(t+1)
        fpl.update_d()

        # calculate flip_pseudo_label_ratio
        if epoch == 0:
            flip_p_label_ratio = nan
            flip_p_label_ratioes.append(flip_p_label_ratio)
            temp_p_label = fpl.d.copy()
        else:
            flip_p_label_ratio = (fpl.d != temp_p_label).mean()
            flip_p_label_ratioes.append(flip_p_label_ratio)
            temp_p_label = fpl.d.copy()

        # train
        used_index = (fpl.d.reshape(-1) != -1)
        train_dataset = DatasetTrain(
            train_data[used_index], train_label[used_index], fpl.d.reshape(-1)[used_index])
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            shuffle=True,  num_workers=cfg.num_workers)

        model.train()
        train_losses = []
        train_gt, train_pseudo, train_pred = [], [], []
        for data, label, pseudo_label in tqdm(train_loader, leave=False):
            data, label = data.to(cfg.device), label.to(cfg.device)
            pseudo_label = pseudo_label.to(cfg.device)

            y = model(data)
            loss = criterion(y, pseudo_label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())
            train_gt.extend(label.cpu().detach().numpy())
            train_pseudo.extend(pseudo_label.cpu().detach().numpy())
            train_pred.extend(y.argmax(1).cpu().detach().numpy())

        train_loss = np.array(train_losses).mean()
        train_loss_list.append(train_loss)

        train_gt = np.array(train_gt)
        train_pseudo = np.array(train_pseudo)
        train_pred = np.array(train_pred)

        train_acc = (train_gt == train_pred).mean()
        train_acc_list.append(train_acc)
        train_pseudo_acc = (train_pseudo == train_pred).mean()
        train_pseudo_acc_list.append(train_pseudo_acc)
        pseudo_label_acc = (train_gt == train_pseudo).mean()
        pseudo_label_acc_list.append(pseudo_label_acc)

        # test
        model.eval()
        test_losses = []
        test_gt, test_pred = [], []
        for data, label in tqdm(test_loader, leave=False):
            data, label = data.to(cfg.device), label.to(cfg.device)
            y = model(data)
            loss = criterion(y, label)
            test_losses.append(loss.item())
            test_gt.extend(label.cpu().detach().numpy())
            test_pred.extend(y.argmax(1).cpu().detach().numpy())
        test_gt, test_pred = np.array(test_gt), np.array(test_pred)
        test_loss = np.array(test_losses).mean()
        test_acc = np.array(test_gt == test_pred).mean()
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        e_time = time.time()

        # print result
        log.info('[Epoch: %d/%d (%ds)] train_loss: %.4f, train_acc: %.4f, train_pseudo_acc: %.4f, flip: %.4f,  test_loss: %.4f, test_acc: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time,
                  train_loss, train_acc, pseudo_label_acc, flip_p_label_ratio,
                  test_loss, test_acc))

        # save
        np.save(result_path+'test_acc', np.array(test_acc_list))
        torch.save(model.state_dict(), result_path+'model.pth')

        plt.plot(train_pseudo_acc_list, label='train_pseudo_acc')
        plt.plot(train_acc_list, label='train_acc')
        plt.plot(test_acc_list, label='test_acc')
        plt.plot(pseudo_label_acc_list, label='label_acc')
        plt.plot(flip_p_label_ratioes, label='flip_label_ratio')
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.savefig(result_path+'acc.png')
        plt.close()


if __name__ == '__main__':
    main()
