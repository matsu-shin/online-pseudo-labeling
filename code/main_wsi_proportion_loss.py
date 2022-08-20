import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import time
import gc
import torchvision.transforms as transforms
from utils import Dataset, fix_seed, make_folder
from load_mnist import load_minist
from load_cifar10 import load_cifar10
from losses import ProportionLoss

log = logging.getLogger(__name__)


class DatasetBagSampling(torch.utils.data.Dataset):
    def __init__(self, bags, label_proportions, num_sampled_instances):
        self.bags = bags
        self.label_proportions = label_proportions
        self.num_sampled_instances = num_sampled_instances
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.len = len(self.bags)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        (b, w, h, c) = self.bags[idx].shape
        if b > self.num_sampled_instances:
            index = np.arange(b)
            sampled_index = np.random.choice(index, self.num_sampled_instances)

            bag = torch.zeros((self.num_sampled_instances, c, w, h))
            for i, j in enumerate(sampled_index):
                bag[i] = self.transform(self.bags[idx][j])
        else:
            bag = torch.zeros((b, c, w, h))
            for i in range(b):
                bag[i] = self.transform(self.bags[idx][i])

        label_proportion = self.label_proportions[idx]
        label_proportion = torch.tensor(label_proportion).float()

        return bag, label_proportion


@ hydra.main(config_path='../config', config_name='config_wsi_proportion_loss')
def main(cfg: DictConfig) -> None:

    # file name
    cwd = hydra.utils.get_original_cwd()
    result_path = cwd + cfg.result_path
    result_path += 'wsi_proportion_loss/'
    make_folder(result_path)
    result_path += '%s' % str(cfg.dataset.name)
    result_path += '_samp_%s' % str(cfg.num_sampled_instances)
    result_path += '_mini_batch_%s' % str(cfg.mini_batch)
    result_path += '/'
    make_folder(result_path)

    fh = logging.FileHandler(result_path+'exec.log')
    log.addHandler(fh)
    log.info(OmegaConf.to_yaml(cfg))
    log.info('cwd:%s' % cwd)

    # load data
    # unlabeled_idx = np.arange(100, 503)
    # unlabeled_idx = np.arange(100, 110)
    # not_used_idx = [124, 134, 261, 270, 353]
    # unlabeled_idx = np.setdiff1d(unlabeled_idx, not_used_idx)

    # dataset_path = '../../../../dataset/WSI/'
    dataset_path = cwd + cfg.dataset.dir

    # with open(dataset_path+"image_name_dict.pkl", "rb") as tf:
    #     image_name_dict = pickle.load(tf)
    with open(dataset_path+"proportion_dict.pkl", "rb") as tf:
        proportion_dict = pickle.load(tf)
    with open(dataset_path+"train_bag_data.pkl", "rb") as tf:
        train_bag_data = pickle.load(tf)

    unlabeled_idx = list(proportion_dict.keys())

    if cfg.dataset.is_debug_labeled:
        with open(dataset_path+"train_data.pkl", "rb") as tf:
            labeled_train_bag_data = pickle.load(tf)
        with open(dataset_path+"train_label.pkl", "rb") as tf:
            labeled_train_bag_label = pickle.load(tf)
        with open(dataset_path+"train_proportion.pkl", "rb") as tf:
            labeled_train_proportion = pickle.load(tf)

        labeled_idx = list(labeled_train_proportion.keys())
        unlabeled_idx += labeled_idx
        train_bag_data.update(labeled_train_bag_data)
        proportion_dict.update(labeled_train_proportion)

    # define model
    fix_seed(cfg.seed)
    fix_seed(cfg.seed)
    if cfg.model == 'resnet50':
        model = resnet50(pretrained=cfg.is_pretrained)
    elif cfg.model == 'resnet18':
        model = resnet18(pretrained=cfg.is_pretrained)
    else:
        log.info('No model!')
    model.fc = nn.Linear(model.fc.in_features, cfg.dataset.num_classes)
    model = model.to(cfg.device)

    # define criterion and optimizer
    loss_function = ProportionLoss(
        metric=cfg.proportion_metric
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # make train_loader
    train_data = []
    for idx in unlabeled_idx:
        train_data.append(train_bag_data[idx])
    # to reduce cpu memory consumption
    del train_bag_data
    gc.collect()
    del labeled_train_bag_data
    gc.collect()

    proportion_dict = list(proportion_dict.values())
    train_dataset = DatasetBagSampling(
        train_data, proportion_dict,
        num_sampled_instances=cfg.num_sampled_instances)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1,
        shuffle=True,  num_workers=cfg.num_workers)

    # make test_loader
    test_data = np.load(dataset_path+'test_data.npy')
    test_label = np.load(dataset_path+'test_label.npy')
    test_dataset = Dataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)

    fix_seed(cfg.seed)
    test_acces = []
    for epoch in range(cfg.num_epochs):
        s_time = time.time()

        # train
        model.train()
        train_losses = []
        flag_optim = 0
        for data, proportion in tqdm(train_loader, leave=False):
            data = data.to(cfg.device)
            proportion = proportion.to(cfg.device)

            (n, b, w, h, c) = data.size()
            data = data.reshape(-1, w, h, c)

            y = model(data)
            confidence = F.softmax(y, dim=1)
            confidence = confidence.reshape(n, b, -1).mean(dim=1)

            loss = loss_function(confidence, proportion) / cfg.mini_batch
            loss.backward()

            flag_optim += 1
            if flag_optim == cfg.mini_batch:
                optimizer.step()
                optimizer.zero_grad()
                flag_optim = 0

            train_losses.append(loss.item())

        train_loss = np.array(train_losses).mean()

        # test
        model.eval()
        test_gt, test_pred = [], []
        for data, label in tqdm(test_loader, leave=False):
            data, label = data.to(cfg.device), label.to(cfg.device)
            y = model(data)
            test_gt.extend(label.cpu().detach().numpy())
            test_pred.extend(y.argmax(1).cpu().detach().numpy())
        test_gt, test_pred = np.array(test_gt), np.array(test_pred)
        test_acc = np.array(test_gt == test_pred).mean()
        test_acces.append(test_acc)

        e_time = time.time()

        # print result
        log.info('[Epoch: %d/%d (%ds)] train_loss: %.4f, test_acc: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time, train_loss, test_acc))

        np.save(result_path+'test_acc', test_acces)
        plt.plot(test_acces, label='test_acc')
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.savefig(result_path+'acc.png')
        plt.close()


if __name__ == '__main__':
    main()
