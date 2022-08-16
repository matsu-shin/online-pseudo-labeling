from cmath import nan
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18
from tqdm import tqdm
from matplotlib import pyplot as plt
from glob import glob
from PIL import Image
import pickle
import time

from FPL_wsi import FPL
from utils import fix_seed, make_folder, save_confusion_matrix

log = logging.getLogger(__name__)


class Dataset_theta(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(data)
        return data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(data)
        label = self.label[idx]
        label = torch.tensor(label).long()
        return data, label


def debug_labeled(model, cfg):
    dataset_path = '../../../../dataset/WSI/'
    train_data = np.load(dataset_path+'train_data.npy')
    train_label = np.load(dataset_path+'train_label.npy')
    train_dataset = Dataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)

    model.eval()
    train_gt, train_pred = [], []
    for data, label in tqdm(train_loader, leave=False):
        data, label = data.to(cfg.device), label.to(cfg.device)
        y = model(data)
        train_gt.extend(label.cpu().detach().numpy())
        train_pred.extend(y.argmax(1).cpu().detach().numpy())
    train_gt, train_pred = np.array(train_gt), np.array(train_pred)
    train_acc = np.array(train_gt == train_pred).mean()
    return train_acc


@ hydra.main(config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:
    fix_seed(cfg.seed)

    # file name
    cwd = hydra.utils.get_original_cwd()
    result_path = cwd+cfg.result_path
    result_path += 'wsi/'
    make_folder(result_path)
    result_path += 'eta_%s' % str(cfg.fpl.eta)
    result_path += '_pseudo_ratio_%s' % str(cfg.pseudo_ratio)
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

    dataset_path = '../../../../dataset/WSI/'

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

    num_instances_dict = {}
    for idx in unlabeled_idx:
        num_instances_dict[idx] = train_bag_data[idx].shape[0]

    train_data = []
    for idx in unlabeled_idx:
        train_data.extend(train_bag_data[idx])
    train_data = np.array(train_data)
    # print(train_data.shape)

    train_dataset_for_fpl = Dataset_theta(train_data)
    train_loader_for_fpl = torch.utils.data.DataLoader(
        train_dataset_for_fpl, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)
    data = next(iter(train_loader_for_fpl))
    # print(data.size())

    # define model, criterion and optimizer
    fix_seed(cfg.seed)
    model = resnet18(pretrained=cfg.is_pretrained)
    model.fc = nn.Linear(model.fc.in_features, cfg.dataset.num_classes)
    model = model.to(cfg.device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # FPL
    fpl = FPL(num_instances_dict=num_instances_dict,
              proportion_dict=proportion_dict,
              sigma=cfg.fpl.sigma,
              eta=cfg.fpl.eta,
              loss_f=cfg.fpl.loss_f,
              pseudo_ratio=cfg.pseudo_ratio,
              is_online_prediction=cfg.fpl.is_online_prediction)

    # make test_loader
    test_data = np.load(dataset_path+'test_data.npy')
    test_label = np.load(dataset_path+'test_label.npy')
    test_dataset = Dataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)

    fix_seed(cfg.seed)
    train_p_acces, test_acces, flip_p_label_ratioes = [], [], []
    p_label_acces, train_acces = [], []
    for epoch in range(cfg.num_epochs):
        s_time = time.time()

        # FPL
        fpl.update_theta(model, train_loader_for_fpl,
                         num_instances_dict, cfg.device)
        fpl.update_d(num_instances_dict)

        if cfg.dataset.is_debug_labeled:
            p_label_acc = []
            for idx in labeled_idx:
                acc_list = fpl.d_dict[idx] == labeled_train_bag_label[idx]
                acc_list = acc_list[fpl.d_dict[idx] != -1]
                p_label_acc.extend(acc_list)
            p_label_acc = np.array(p_label_acc).mean()
            p_label_acces.append(p_label_acc)

        # flatten pseudo label
        p_label = []
        for idx in unlabeled_idx:
            p_label.extend(fpl.d_dict[idx])
        p_label = np.array(p_label)

        # calculate flip_pseudo_label_ratio
        if epoch == 0:
            flip_p_label_ratio = nan
            flip_p_label_ratioes.append(flip_p_label_ratio)
            temp_p_label = p_label.copy()
        else:
            flip_p_label_ratio = (p_label != temp_p_label).mean()
            flip_p_label_ratioes.append(flip_p_label_ratio)
            temp_p_label = p_label.copy()

        # train
        train_dataset = Dataset(
            train_data[p_label != -1], p_label[p_label != -1])
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            shuffle=True,  num_workers=cfg.num_workers)
        data, p_label = next(iter(train_loader))
        # print(data.size(), p_label.size())

        model.train()
        train_losses = []
        train_gt, train_pred = [], []
        for data, p_label in tqdm(train_loader, leave=False):
            data, p_label = data.to(cfg.device), p_label.to(cfg.device)
            y = model(data)
            loss = loss_function(y, p_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())
            train_gt.extend(p_label.cpu().detach().numpy())
            train_pred.extend(y.argmax(1).cpu().detach().numpy())
        train_gt, train_pred = np.array(train_gt), np.array(train_pred)
        train_loss = np.array(train_losses).mean()
        train_p_acc = np.array(train_gt == train_pred).mean()
        train_p_acces.append(train_p_acc)

        # test
        if cfg.dataset.is_debug_labeled:
            train_acc = debug_labeled(model, cfg)
            train_acces.append(train_acc)
            log.info('pseudo_label_acc: %.4f, train_acc: %.4f' %
                     (p_label_acc, train_acc))

        model.eval()
        test_losses = []
        test_gt, test_pred = [], []
        for data, label in tqdm(test_loader, leave=False):
            data, label = data.to(cfg.device), label.to(cfg.device)
            y = model(data)
            loss = loss_function(y, label)
            test_losses.append(loss.item())
            test_gt.extend(label.cpu().detach().numpy())
            test_pred.extend(y.argmax(1).cpu().detach().numpy())
        test_gt, test_pred = np.array(test_gt), np.array(test_pred)
        test_loss = np.array(test_losses).mean()
        test_acc = np.array(test_gt == test_pred).mean()
        test_acces.append(test_acc)

        e_time = time.time()
        log.info('[Epoch: %d/%d (%ds)] train_loss: %.4f, train_p_acc: %.4f, flip: %.4f, test_loss: %.4f, test_acc: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time, train_loss, train_p_acc, flip_p_label_ratio, test_loss, test_acc))
        save_confusion_matrix(gt=test_gt, pred=test_pred,
                              path=result_path+'cm.png', epoch=epoch+1)

        np.save(result_path+'train_acc', train_acces)
        np.save(result_path+'train_pseudo_acc', train_p_acces)
        np.save(result_path+'test_acc', test_acces)
        np.save(result_path+'flip_pseudo_label_ratio', flip_p_label_ratioes)
        np.save(result_path+'pseudo_label_acc', p_label_acces)

        plt.plot(train_acces, label='train_acc')
        plt.plot(train_p_acces, label='train_pseudo_acc')
        plt.plot(test_acces, label='test_acc')
        plt.plot(flip_p_label_ratioes, label='flip_pseudo_label_ratio')
        plt.plot(p_label_acces, label='pseudo_label_acc')
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.savefig(result_path+'acc.png')
        plt.close()

        # save
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), result_path +
                       'model_'+str(epoch+1)+'.pth')
            save_confusion_matrix(gt=test_gt, pred=test_pred,
                                  path=result_path+'cm_'+str(epoch+1)+'.png', epoch=epoch+1)


if __name__ == '__main__':
    main()
