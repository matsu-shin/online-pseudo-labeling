from math import nan
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

from FPL import FPL
from losses import ProportionLoss
from utils import Dataset, fix_seed, make_folder, save_confusion_matrix
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


class DatasetBagSampling(torch.utils.data.Dataset):
    def __init__(self, bags, labels, label_proportions):
        self.bags = bags
        self.labels = labels
        self.label_proportions = label_proportions
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        (self.n, self.b, self.w, self.h, self.c) = self.bags.shape
        self.len = self.n

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        bag = torch.zeros((self.b, self.c, self.w, self.h))
        label = torch.zeros(self.b)
        for i in range(self.b):
            bag[i] = self.transform(self.bags[idx][i])
            label[i] = self.labels[idx][i]

        label_proportion = self.label_proportions[idx]
        label_proportion = torch.tensor(label_proportion).float()

        return bag, label, label_proportion


@ hydra.main(config_path='../config', config_name='config_toy_fpl')
def main(cfg: DictConfig) -> None:
    fix_seed(cfg.seed)

    # file name
    cwd = hydra.utils.get_original_cwd()
    result_path = cwd + cfg.result_path
    result_path += 'toy_fpl/'
    make_folder(result_path)
    result_path += '%s' % (cfg.dataset.name)
    if cfg.fpl.is_online_prediction == False:
        result_path += '_not_op_'

    # result_path += 'p_ratio_%s' % (cfg.pseudo_ratio)
    result_path += '_%s' % (cfg.fpl.loss_f)
    result_path += '_eta_%s' % (cfg.fpl.eta)
    result_path += '_lr_%s' % (cfg.lr)
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
    log.info('loading...  '+dataset_path)
    bags_index = np.load(dataset_path)
    num_classes = cfg.dataset.num_classes

    # make validation
    n_val = int(bags_index.shape[0]*cfg.validation)
    val_bags_index = bags_index[: n_val]
    bags_index = bags_index[n_val:]

    val_bags_data, val_bags_label = train_data[val_bags_index], train_label[val_bags_index]
    val_label_proportion = np.identity(
        num_classes)[val_bags_label].mean(axis=1)
    val_dataset = DatasetBagSampling(
        val_bags_data, val_bags_label, val_label_proportion)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.mini_batch,
        shuffle=False,  num_workers=cfg.num_workers)

    test_dataset = Dataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)

    # define model
    fix_seed(cfg.seed)
    if cfg.model == 'resnet50':
        model = resnet50(pretrained=cfg.is_pretrained)
    elif cfg.model == 'resnet18':
        model = resnet18(pretrained=cfg.is_pretrained)
    else:
        log.info('No model!')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(cfg.device)

    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    proportion_loss = ProportionLoss(
        metric=cfg.proportion_metric
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # fpl
    bags_data, bags_label = train_data[bags_index], train_label[bags_index]
    label_proportion = np.identity(num_classes)[bags_label].mean(axis=1)
    num_bags = bags_index.shape[0]
    num_instances = bags_index.shape[1]

    fpl = FPL(label_proportion=label_proportion,
              num_bags=num_bags, num_instances=num_instances,
              sigma=cfg.fpl.sigma,
              eta=cfg.fpl.eta,
              loss_f=cfg.fpl.loss_f,
              pseudo_ratio=cfg.pseudo_ratio,
              is_online_prediction=cfg.fpl.is_online_prediction)

    (_, _, w, h, c) = bags_data.shape
    train_data = bags_data.reshape(-1, w, h, c)
    train_label = bags_label.reshape(-1)
    train_dataset_for_fpl = Dataset(train_data, train_label)
    train_loader_for_fpl = torch.utils.data.DataLoader(
        train_dataset_for_fpl, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)

    train_acces, val_acces, test_acces = [], [], []
    train_pseudo_acces = []
    pseudo_label_acces = []
    flip_p_label_ratioes = []
    train_losses, val_losses, test_losses = [], [], []
    best_validation_loss = np.inf
    final_acc = 0

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
        losses = []
        gt, pseudo, pred = [], [], []
        for data, label, pseudo_label in tqdm(train_loader, leave=False):
            data, label = data.to(cfg.device), label.to(cfg.device)
            pseudo_label = pseudo_label.to(cfg.device)

            y = model(data)
            loss = criterion(y, pseudo_label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            gt.extend(label.cpu().detach().numpy())
            pseudo.extend(pseudo_label.cpu().detach().numpy())
            pred.extend(y.argmax(1).cpu().detach().numpy())

        train_loss = np.array(losses).mean()
        train_losses.append(train_loss)

        gt, pseudo, pred = np.array(gt), np.array(pseudo), np.array(pred)
        train_acc = (gt == pred).mean()
        pseudo_acc = (pseudo == pred).mean()
        pseudo_label_acc = (gt == pseudo).mean()
        train_cm = confusion_matrix(y_true=gt, y_pred=pred)
        train_pseudo_cm = confusion_matrix(y_true=pseudo, y_pred=pred)
        pseudo_label_cm = confusion_matrix(y_true=gt, y_pred=pseudo)
        train_acces.append(train_acc)
        train_pseudo_acces.append(pseudo_acc)
        pseudo_label_acces.append(pseudo_label_acc)

        e_time = time.time()
        log.info('[Epoch: %d/%d (%ds)] train_loss: %.4f, train_acc: %.4f, train_pseudo_acc: %.4f, pseudo_label_acc: %.4f, flip:  %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time, train_loss, train_acc, pseudo_acc, pseudo_label_acc, flip_p_label_ratio))

        # validation
        s_time = time.time()
        model.eval()
        losses = []
        gt, pred = [], []
        with torch.no_grad():
            for data, label, proportion in tqdm(val_loader, leave=False):
                data, label = data.to(cfg.device), label.to(cfg.device)
                proportion = proportion.to(cfg.device)

                (n, b, w, h, c) = data.size()
                data = data.reshape(-1, w, h, c)
                label = label.reshape(-1)

                y = model(data)
                confidence = F.softmax(y, dim=1)
                confidence = confidence.reshape(n, b, -1).mean(dim=1)
                loss = proportion_loss(confidence, proportion)

                gt.extend(label.cpu().detach().numpy())
                pred.extend(y.argmax(1).cpu().detach().numpy())
                losses.append(loss.item())

        val_loss = np.array(losses).mean()
        val_acc = (np.array(gt) == np.array(pred)).mean()
        val_losses.append(val_loss)
        val_acces.append(val_acc)
        log.info('[Epoch: %d/%d (%ds)] val_loss: %.4f, val_acc: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time, val_loss, val_acc))

        # test
        s_time = time.time()
        model.eval()
        losses = []
        gt, pred = [], []
        with torch.no_grad():
            for data, label in tqdm(test_loader, leave=False):
                data, label = data.to(cfg.device), label.to(cfg.device)
                y = model(data)
                loss = criterion(y, label)
                losses.append(loss.item())
                gt.extend(label.cpu().detach().numpy())
                pred.extend(y.argmax(1).cpu().detach().numpy())
        gt, pred = np.array(gt), np.array(pred)
        test_loss = np.array(losses).mean()
        test_acc = np.array(gt == pred).mean()
        test_cm = confusion_matrix(y_true=gt, y_pred=pred)
        test_losses.append(test_loss)
        test_acces.append(test_acc)

        e_time = time.time()

        # print result
        log.info('[Epoch: %d/%d (%ds)] test_loss: %.4f, test_acc: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time,
                  test_loss, test_acc))
        log.info('--------------------------------------------------')

        if val_loss < best_validation_loss:
            torch.save(model.state_dict(), result_path + 'best_model.pth')
            save_confusion_matrix(cm=train_cm, path=result_path+'cm_train.png',
                                  title='epoch: %d, train_acc: %.4f' % (epoch+1, train_acc))
            save_confusion_matrix(cm=train_pseudo_cm, path=result_path+'cm_train_pseudo.png',
                                  title='epoch: %d, train_pesudo_acc: %.4f' % (epoch+1, pseudo_acc))
            save_confusion_matrix(cm=pseudo_label_cm, path=result_path+'cm_pseudo_label.png',
                                  title='epoch: %d, pseudo_label_acc: %.4f' % (epoch+1, pseudo_label_acc))
            save_confusion_matrix(cm=test_cm, path=result_path+'cm_test.png',
                                  title='epoch: %d, test_acc: %.4f' % (epoch+1, test_acc))

            best_validation_loss = val_loss
            final_acc = test_acc

        # if (epoch+1)%10==0:
        torch.save(model.state_dict(), result_path +
                   'model_%d.pth' % (epoch+1))

        np.save(result_path+'train_acc', train_acces)
        np.save(result_path+'val_acc', val_acces)
        np.save(result_path+'test_acc', test_acces)
        np.save(result_path+'train_pseudo_acc', train_pseudo_acces)
        np.save(result_path+'pseudo_label_acc', pseudo_label_acces)
        np.save(result_path+'flip_p_label_ratio', flip_p_label_ratioes)
        plt.plot(train_acces, label='train_acc')
        plt.plot(val_acces, label='val_acc')
        plt.plot(test_acces, label='test_acc')
        plt.plot(train_pseudo_acces, label='train_pseudo_acc')
        plt.plot(pseudo_label_acces, label='pseudo_label_acc')
        plt.plot(flip_p_label_ratioes, label='flip_p_label_ratio')
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.savefig(result_path+'acc.png')
        plt.close()

        np.save(result_path+'train_loss', train_losses)
        np.save(result_path+'val_loss', val_losses)
        np.save(result_path+'test_loss', test_losses)
        plt.plot(train_losses, label='train_loss')
        plt.plot(val_losses, label='val_loss')
        plt.plot(test_losses, label='test_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(result_path+'loss.png')
        plt.close()

    log.info(OmegaConf.to_yaml(cfg))
    log.info('final_acc: %.4f' % (final_acc))
    log.info('--------------------------------------------------')


if __name__ == '__main__':
    main()
