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
    def __init__(self, data_path):
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = len(self.data_path)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = Image.open(self.data_path[idx])
        data = np.asarray(data.convert('RGB'))
        data = self.transform(data)
        return data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label):
        self.data_path = data_path
        self.label = label
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = len(self.data_path)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = Image.open(self.data_path[idx])
        data = np.asarray(data.convert('RGB'))
        data = self.transform(data)

        label = self.label[idx]
        label = torch.tensor(label).long()

        return data, label


class DatasetTest(torch.utils.data.Dataset):
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


@ hydra.main(config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:
    fix_seed(cfg.seed)

    # file name
    cwd = hydra.utils.get_original_cwd()
    result_path = cwd+cfg.result_path
    result_path += 'wsi_load/'
    # make folder
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
    with open(dataset_path+"image_name_dict.pkl", "rb") as tf:
        image_name_dict = pickle.load(tf)
    with open(dataset_path+"proportion_dict.pkl", "rb") as tf:
        proportion_dict = pickle.load(tf)
    # with open(dataset_path+"train_bag_data.pkl", "rb") as tf:
    #     train_bag_data = pickle.load(tf)
    unlabeled_idx = list(proportion_dict.keys())
    train_bag_data_path = {}
    for idx in unlabeled_idx:
        wsi_name = image_name_dict[idx]
        instance_path_list = glob(dataset_path+'unlabeled/'+wsi_name+'/_/*')
        train_bag_data_path[idx] = instance_path_list

    num_instances_dict = {}
    for idx in unlabeled_idx:
        num_instances_dict[idx] = len(train_bag_data_path[idx])

    train_data_path = []
    for idx in unlabeled_idx:
        train_data_path.extend(train_bag_data_path[idx])

    train_dataset_for_fpl = Dataset_theta(train_data_path)
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
              is_online_prediction=cfg.fpl.is_online_prediction)

    # make test_loader
    test_data = np.load(dataset_path+'test_data.npy')
    test_label = np.load(dataset_path+'test_label.npy')
    test_dataset = DatasetTest(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)

    fix_seed(cfg.seed)
    train_acces, test_acces, flip_p_label_ratioes = [], [], []
    for epoch in range(cfg.num_epochs):
        s_time = time.time()

        # FPL
        fpl.update_theta(model, train_loader_for_fpl,
                         num_instances_dict, cfg.device)
        fpl.update_d(num_instances_dict)

        p_label = []
        for idx in unlabeled_idx:
            p_label.extend(fpl.d_dict[idx])
        p_label = np.array(p_label)

        if epoch == 0:
            flip_p_label_ratio = nan
            flip_p_label_ratioes.append(flip_p_label_ratio)
            temp_p_label = p_label.copy()
        else:
            flip_p_label_ratio = (p_label != temp_p_label).mean()
            flip_p_label_ratioes.append(flip_p_label_ratio)
            temp_p_label = p_label.copy()
        # print(pseudo_label.shape)

        # train
        train_dataset = Dataset(train_data_path, p_label)
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
        train_acc = np.array(train_gt == train_pred).mean()
        train_acces.append(train_acc)

        # test
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
        log.info('[Epoch: %d/%d (%ds)] train_loss: %.4f, train_acc: %.4f, flip: %.4f, test_loss: %.4f, test_acc: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time, train_loss, train_acc, flip_p_label_ratio, test_loss, test_acc))
        save_confusion_matrix(gt=test_gt, pred=test_pred,
                              path=result_path+'cm.png', epoch=epoch+1)

        plt.plot(train_acces, label='train_acc')
        plt.plot(test_acces, label='test_acc')
        plt.plot(flip_p_label_ratioes, label='flip_pseudo_label_ratio')
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
