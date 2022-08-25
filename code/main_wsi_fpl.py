from math import nan
from unittest import result
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
from tqdm import tqdm
from matplotlib import pyplot as plt
from glob import glob
from PIL import Image
import pickle
import time
import gc
from sklearn.metrics import confusion_matrix

from FPL_wsi import FPL
from utils import Dataset, cal_OP_PC_mIoU, fix_seed, make_folder, save_confusion_matrix
from losses import ProportionLoss

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


class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, data, label, used_index):
        self.data = data
        self.label = label
        self.used_index = used_index
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = self.used_index.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        index = self.used_index[idx]
        data = self.data[index]
        data = self.transform(data)
        label = self.label[index]
        label = torch.tensor(label).long()
        return data, label


class DatasetBagSampling(torch.utils.data.Dataset):
    def __init__(self, bags, label_proportions):
        self.bags = bags
        self.label_proportions = label_proportions
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = len(self.bags)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        (b, w, h, c) = self.bags[idx].shape
        bag = torch.zeros((b, c, w, h))
        for i in range(b):
            bag[i] = self.transform(self.bags[idx][i])

        label_proportion = self.label_proportions[idx]
        label_proportion = torch.tensor(label_proportion).float()

        return bag, label_proportion


def debug_labeled(model, dataset_path, cfg):
    train_data = np.load(dataset_path+'train_data.npy')
    train_label = np.load(dataset_path+'train_label.npy')
    train_dataset = Dataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)

    model.eval()
    with torch.no_grad():
        gt, pred = [], []
        for data, label in tqdm(train_loader, leave=False):
            data, label = data.to(cfg.device), label.to(cfg.device)
            y = model(data)
            gt.extend(label.cpu().detach().numpy())
            pred.extend(y.argmax(1).cpu().detach().numpy())
        gt, pred = np.array(gt), np.array(pred)

    return gt, pred


@ hydra.main(config_path='../config', config_name='config_wsi_fpl')
def main(cfg: DictConfig) -> None:
    fix_seed(cfg.seed)

    # file name
    cwd = hydra.utils.get_original_cwd()
    result_path = cwd + cfg.result_path
    result_path += 'wsi_fpl/'
    make_folder(result_path)
    if cfg.fpl.is_online_prediction == False:
        result_path += 'not_op_'
    result_path += '%s' % (cfg.fpl.loss_f)
    result_path += '_eta_%s' % str(cfg.fpl.eta)
    result_path += '_lr_%s' % str(cfg.lr)
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

    n_val = int(len(unlabeled_idx)*cfg.validation)
    fix_seed(cfg.seed)
    np.random.shuffle(unlabeled_idx)
    val_idx = unlabeled_idx[: n_val]
    log.info(val_idx)
    unlabeled_idx = unlabeled_idx[n_val:]

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

    # make val_loader
    num_instances_dict = {}
    for idx in unlabeled_idx:
        num_instances_dict[idx] = train_bag_data[idx].shape[0]

    train_data = []
    for idx in unlabeled_idx:
        train_data.extend(train_bag_data[idx])
    train_data = np.array(train_data)

    val_data, val_proportion = [], []
    for idx in val_idx:
        val_data.append(train_bag_data[idx])
        val_proportion.append(proportion_dict[idx])

    val_dataset = DatasetBagSampling(val_data, val_proportion)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        shuffle=True,  num_workers=cfg.num_workers)

    # to reduce cpu memory consumption
    del train_bag_data
    gc.collect()
    del labeled_train_bag_data
    gc.collect()

    # FPL
    fpl = FPL(num_instances_dict=num_instances_dict,
              proportion_dict=proportion_dict,
              sigma=cfg.fpl.sigma,
              eta=cfg.fpl.eta,
              loss_f=cfg.fpl.loss_f,
              pseudo_ratio=cfg.pseudo_ratio,
              is_online_prediction=cfg.fpl.is_online_prediction)

    train_dataset_for_fpl = Dataset_theta(train_data)
    train_loader_for_fpl = torch.utils.data.DataLoader(
        train_dataset_for_fpl, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)
    # data = next(iter(train_loader_for_fpl))
    # print(data.size())

    # make test_loader
    test_data = np.load(dataset_path+'test_data.npy')
    test_label = np.load(dataset_path+'test_label.npy')
    test_dataset = Dataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)

    # define model, criterion and optimizer
    fix_seed(cfg.seed)
    if cfg.model == 'resnet50':
        model = resnet50(pretrained=cfg.is_pretrained)
    elif cfg.model == 'resnet18':
        model = resnet18(pretrained=cfg.is_pretrained)
    else:
        log.info('No model!')
    model.fc = nn.Linear(model.fc.in_features, cfg.dataset.num_classes)
    model = model.to(cfg.device)

    weight = 1 / torch.tensor(list(fpl.k_dict.values())).float().sum(dim=0)
    weight = weight.to(cfg.device)
    loss_function = nn.CrossEntropyLoss(weight=weight)
    proportion_loss = ProportionLoss(metric=cfg.proportion_metric)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    fix_seed(cfg.seed)
    train_OPs, train_PCs, train_mIoUs = [], [], []
    train_pseudo_OPs, train_pseudo_PCs, train_pseudo_mIoUs = [], [], []
    test_OPs, test_PCs, test_mIoUs = [], [], []
    pseudo_label_acces = []
    flip_p_label_ratioes = []
    train_losses, val_losses = [], []
    best_validation_loss = np.inf
    for epoch in range(cfg.num_epochs):
        # FPL
        s_time = time.time()

        fpl.update_theta(model, train_loader_for_fpl,
                         num_instances_dict, cfg.device)
        fpl.update_d(num_instances_dict)

        if cfg.dataset.is_debug_labeled:
            p_label_acc = []
            for idx in labeled_idx:
                acc_list = fpl.d_dict[idx] == labeled_train_bag_label[idx]
                acc_list = acc_list[fpl.d_dict[idx] != -1]
                p_label_acc.extend(acc_list)
            pseudo_label_acc = np.array(p_label_acc).mean()
            pseudo_label_acces.append(pseudo_label_acc)

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
        # to reduce cpu memory consumption
        train_dataset = DatasetTrain(train_data, p_label,
                                     np.where(p_label != -1)[0])
        # train_dataset = Dataset(
        #     train_data[p_label != -1], p_label[p_label != -1])
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            shuffle=True,  num_workers=cfg.num_workers)
        # data, p_label = next(iter(train_loader))
        # print(data.size(), p_label.size())

        model.train()
        losses = []
        gt, pred = [], []
        for data, p_label in tqdm(train_loader, leave=False):
            data, p_label = data.to(cfg.device), p_label.to(cfg.device)
            y = model(data)
            loss = loss_function(y, p_label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            gt.extend(p_label.cpu().detach().numpy())
            pred.extend(y.argmax(1).cpu().detach().numpy())

        train_loss = np.array(losses).mean()
        gt, pred = np.array(gt), np.array(pred)

        train_pseudo_cm = confusion_matrix(y_true=gt, y_pred=pred)
        train_pseudo_OP, train_pseudo_PC, train_pseudo_mIoU = cal_OP_PC_mIoU(
            train_pseudo_cm)
        train_pseudo_OPs.append(train_pseudo_OP)
        train_pseudo_PCs.append(train_pseudo_PC)
        train_pseudo_mIoUs.append(train_pseudo_mIoU)

        if cfg.dataset.is_debug_labeled:
            gt, pred = debug_labeled(
                model=model, dataset_path=dataset_path, cfg=cfg)
            train_cm = confusion_matrix(y_true=gt, y_pred=pred)
            train_OP, train_PC, train_mIoU = cal_OP_PC_mIoU(train_cm)
            train_OPs.append(train_OP)
            train_PCs.append(train_PC)
            train_mIoUs.append(train_mIoU)

        e_time = time.time()
        log.info('[Epoch: %d/%d (%ds)] train_loss: %.4f, p_OP: %.4f, p_PC: %.4f, p_mIoU: %.4f, OP: %.4f, PC: %.4f, mIoU: %.4f, pseudo_label_acc: %.4f, flip:  %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time, train_loss,
                  train_pseudo_OP, train_pseudo_PC, train_pseudo_mIoU,
                  train_OP, train_PC, train_mIoU,
                  pseudo_label_acc, flip_p_label_ratio))

        # validation
        s_time = time.time()
        model.eval()
        losses = []
        with torch.no_grad():
            for data, proportion in tqdm(val_loader, leave=False):
                data, proportion = data[0], proportion[0]

                confidence = []
                if (data.size(0) % cfg.batch_size) == 0:
                    J = int((data.size(0)//cfg.batch_size))
                else:
                    J = int((data.size(0)//cfg.batch_size)+1)

                for j in range(J):
                    if j+1 != J:
                        data_j = data[j*cfg.batch_size: (j+1)*cfg.batch_size]
                    else:
                        data_j = data[j*cfg.batch_size:]

                    data_j = data_j.to(cfg.device)
                    y = model(data_j)
                    confidence.extend(
                        F.softmax(y, dim=1).cpu().detach().numpy())

                pred = torch.tensor(confidence).mean(dim=0)
                prop_loss = proportion_loss(pred, proportion)
                losses.append(prop_loss.item())

        val_loss = np.array(losses).mean()
        val_losses.append(val_loss)
        e_time = time.time()
        log.info('[Epoch: %d/%d (%ds)] val_loss: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time, val_loss))

        # test
        s_time = time.time()
        model.eval()
        losses = []
        gt, pred = [], []
        with torch.no_grad():
            for data, label in tqdm(test_loader, leave=False):
                data, label = data.to(cfg.device), label.to(cfg.device)
                y = model(data)
                loss = loss_function(y, label)
                losses.append(loss.item())
                gt.extend(label.cpu().detach().numpy())
                pred.extend(y.argmax(1).cpu().detach().numpy())

        test_loss = np.array(losses).mean()
        gt, pred = np.array(gt), np.array(pred)

        test_cm = confusion_matrix(y_true=gt, y_pred=pred)
        test_OP, test_PC, test_mIoU = cal_OP_PC_mIoU(test_cm)
        test_OPs.append(test_OP)
        test_PCs.append(test_PC)
        test_mIoUs.append(test_mIoU)

        e_time = time.time()

        log.info('[Epoch: %d/%d (%ds)] test loss: %.4f, OP: %.4f, PC: %.4f, mIoU: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time,
                  test_loss, test_OP, test_PC, test_mIoU))
        log.info(
            '-----------------------------------------------------------------------')

        if val_loss < best_validation_loss:
            torch.save(model.state_dict(), result_path + 'best_model.pth')
            save_confusion_matrix(cm=train_cm, path=result_path+'cm_train.png',
                                  title='epoch: %d, OP: %.4f, PC: %.4f, mIoU: %.4f' % (epoch+1, train_OP, train_PC, train_mIoU))
            save_confusion_matrix(cm=test_cm, path=result_path+'cm_test.png',
                                  title='epoch: %d, OP: %.4f, PC: %.4f, mIoU: %.4f' % (epoch+1, test_OP, test_PC, test_mIoU))

            best_validation_loss = val_loss
            final_OP = test_OP
            final_PC = test_PC
            final_mIoU = test_mIoU

        # if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), result_path +
                   'model_%d.pth' % (epoch+1))

        np.save(result_path+'train_OP', train_OPs)
        np.save(result_path+'train_PC', train_PCs)
        np.save(result_path+'train_mIoU', train_mIoUs)
        np.save(result_path+'train_pseudo_OP', train_pseudo_OPs)
        np.save(result_path+'train_pseudo_PC', train_pseudo_PCs)
        np.save(result_path+'train_pseudo_mIoU', train_pseudo_mIoUs)
        np.save(result_path+'test_OP', test_OPs)
        np.save(result_path+'test_PC', test_PCs)
        np.save(result_path+'test_mIoU', test_mIoUs)

        np.save(result_path+'flip_pseudo_label_ratio', flip_p_label_ratioes)
        np.save(result_path+'pseudo_label_acc', pseudo_label_acces)

        plt.plot(train_OPs, label='train_OP')
        plt.plot(train_PCs, label='train_PC')
        plt.plot(train_mIoUs, label='train_mIoU')
        plt.plot(test_OPs, label='test_OP')
        plt.plot(test_PCs, label='test_PC')
        plt.plot(test_mIoUs, label='test_mIoU')
        plt.plot(flip_p_label_ratioes, label='flip_pseudo_label_ratio')
        plt.plot(pseudo_label_acces, label='pseudo_label_acc')
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.savefig(result_path+'curve_acc.png')
        plt.close()

        np.save(result_path+'train_loss', train_losses)
        np.save(result_path+'val_loss', val_losses)
        plt.plot(train_losses, label='train_loss')
        plt.plot(val_losses, label='val_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(result_path+'curve_loss.png')
        plt.close()

    log.info(OmegaConf.to_yaml(cfg))
    log.info('OP: %.4f, PC: %.4f, mIoU: %.4f' %
             (final_OP, final_PC, final_mIoU))
    log.info('--------------------------------------------------')


if __name__ == '__main__':
    main()
