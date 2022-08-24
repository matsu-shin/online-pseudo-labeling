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
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from utils import Dataset, fix_seed, make_folder, get_rampup_weight, cal_OP_PC_mIoU, save_confusion_matrix
from load_mnist import load_minist
from load_cifar10 import load_cifar10
from losses import PiModelLoss, ProportionLoss, VATLoss

log = logging.getLogger(__name__)


class DatasetBagSampling(torch.utils.data.Dataset):
    def __init__(self, bags, label_proportions, num_sampled_instances):
        self.bags = bags
        self.label_proportions = label_proportions
        self.num_sampled_instances = num_sampled_instances
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
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
    if cfg.consistency != 'none':
        result_path += cfg.consistency
    result_path += '_samp_%s' % str(cfg.num_sampled_instances)
    result_path += '_mini_batch_%s' % str(cfg.mini_batch)
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

    # define model
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
    if cfg.consistency == 'none':
        consistency_criterion = None
    elif cfg.consistency == 'vat':
        consistency_criterion = VATLoss()
    elif cfg.consistency == 'pi':
        consistency_criterion = PiModelLoss()
    else:
        raise NameError('Unknown consistency criterion')

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # make train_loader, val_loader
    train_data, train_proportion = [], []
    for idx in unlabeled_idx:
        train_data.append(train_bag_data[idx])
        train_proportion.append(proportion_dict[idx])
    val_data, val_proportion = [], []
    for idx in val_idx:
        val_data.append(train_bag_data[idx])
        val_proportion.append(proportion_dict[idx])
    # to reduce cpu memory consumption
    del train_bag_data
    gc.collect()

    train_dataset = DatasetBagSampling(
        train_data, train_proportion,
        num_sampled_instances=cfg.num_sampled_instances)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1,
        shuffle=True,  num_workers=cfg.num_workers)

    val_dataset = DatasetBagSampling(
        val_data, val_proportion,
        num_sampled_instances=cfg.num_sampled_instances)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        shuffle=True,  num_workers=cfg.num_workers)

    # make train_acc_loader, test_loader
    labeled_train_data, labeled_train_label = [], []
    for idx in labeled_idx:
        labeled_train_data.extend(labeled_train_bag_data[idx])
        labeled_train_label.extend(labeled_train_bag_label[idx])
    labeled_train_data = np.array(labeled_train_data)
    labeled_train_label = np.array(labeled_train_label)
    train_acc_dataset = Dataset(labeled_train_data, labeled_train_label)
    train_acc_loader = torch.utils.data.DataLoader(
        train_acc_dataset, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)
    del labeled_train_bag_data
    gc.collect()

    test_data = np.load(dataset_path+'test_data.npy')
    test_label = np.load(dataset_path+'test_label.npy')
    test_dataset = Dataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)

    fix_seed(cfg.seed)
    # train_acces, test_acces = [], []
    train_OPs, train_PCs, train_mIoUs = [], [], []
    test_OPs, test_PCs, test_mIoUs = [], [], []
    train_losses, val_losses = [], []
    best_validation_loss = np.inf
    final_OP, final_PC, final_mIoU = 0, 0, 0
    for epoch in range(cfg.num_epochs):
        s_time = time.time()

        # train
        model.train()
        losses = []
        b_list = [0]
        mb_data, mb_proportion = [], []
        for iteration, (data, proportion) in enumerate(tqdm(train_loader, leave=False)):
            data = data[0]
            b_list.append(b_list[-1]+data.size(0))
            mb_data.extend(data)
            mb_proportion.extend(proportion)

            if (iteration+1) % cfg.mini_batch == 0 or (iteration + 1) == len(train_loader):
                mb_data = torch.stack(mb_data)
                mb_proportion = torch.stack(mb_proportion)
                mb_data = mb_data.to(cfg.device)
                mb_proportion = mb_proportion.to(cfg.device)

                if cfg.consistency == "vat":
                    # VAT should be calculated before the forward for cross entropy
                    consistency_loss = consistency_criterion(model, mb_data)
                elif cfg.consistency == "pi":
                    consistency_loss, _ = consistency_criterion(model, mb_data)
                else:
                    consistency_loss = torch.tensor(0.)
                alpha = get_rampup_weight(0.05, iteration, -1)
                consistency_loss = alpha * consistency_loss

                y = model(mb_data)
                confidence = F.softmax(y, dim=1)
                pred = torch.zeros(mb_proportion.size(
                    0), cfg.dataset.num_classes).to(cfg.device)
                for n in range(mb_proportion.size(0)):
                    pred[n] = torch.mean(
                        confidence[b_list[n]: b_list[n+1]], dim=0)
                prop_loss = loss_function(pred, mb_proportion)

                loss = prop_loss + consistency_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                b_list = [0]
                mb_data, mb_proportion = [], []

                losses.append(loss.item())

        train_loss = np.array(losses).mean()
        train_losses.append(train_loss)

        # cal train acc
        model.eval()
        gt, pred = [], []
        with torch.no_grad():
            for data, label in tqdm(train_acc_loader, leave=False):
                data, label = data.to(cfg.device), label.to(cfg.device)
                y = model(data)
                gt.extend(label.cpu().detach().numpy())
                pred.extend(y.argmax(1).cpu().detach().numpy())
        train_acc = (np.array(gt) == np.array(pred)).mean()

        train_cm = confusion_matrix(y_true=gt, y_pred=pred)
        train_OP, train_PC, train_mIoU = cal_OP_PC_mIoU(train_cm)
        train_OPs.append(train_OP)
        train_PCs.append(train_PC)
        train_mIoUs.append(train_mIoU)
        log.info('[Epoch: %d/%d] train_loss: %.4f, train acc: %.4f, OP: %.4f, PC: %.4f, mIoU: %.4f' %
                 (epoch+1, cfg.num_epochs,
                  train_loss, train_acc, train_OP, train_PC, train_mIoU))

        # validation
        model.eval()
        losses = []
        b_list = [0]
        mb_data, mb_proportion = [], []
        with torch.no_grad():
            for iteration, (data, proportion) in enumerate(tqdm(val_loader, leave=False)):
                data = data[0]
                b_list.append(b_list[-1]+data.size(0))
                mb_data.extend(data)
                mb_proportion.extend(proportion)

                if (iteration+1) % cfg.mini_batch == 0 or (iteration + 1) == len(val_loader):
                    mb_data = torch.stack(mb_data)
                    mb_proportion = torch.stack(mb_proportion)
                    mb_data = mb_data.to(cfg.device)
                    mb_proportion = mb_proportion.to(cfg.device)

                    y = model(mb_data)
                    confidence = F.softmax(y, dim=1)
                    pred = torch.zeros(mb_proportion.size(
                        0), cfg.dataset.num_classes).to(cfg.device)
                    for n in range(mb_proportion.size(0)):
                        pred[n] = torch.mean(
                            confidence[b_list[n]: b_list[n+1]], dim=0)
                    prop_loss = loss_function(pred, mb_proportion)

                    b_list = [0]
                    mb_data, mb_proportion = [], []

                    losses.append(prop_loss.item())

        val_loss = np.array(losses).mean()
        val_losses.append(val_loss)
        log.info('[Epoch: %d/%d] val_loss: %.4f' %
                 (epoch+1, cfg.num_epochs, val_loss))

        # test
        model.eval()
        gt, pred = [], []
        with torch.no_grad():
            for data, label in tqdm(test_loader, leave=False):
                data, label = data.to(cfg.device), label.to(cfg.device)
                y = model(data)
                gt.extend(label.cpu().detach().numpy())
                pred.extend(y.argmax(1).cpu().detach().numpy())
        test_acc = (np.array(gt) == np.array(pred)).mean()

        test_cm = confusion_matrix(y_true=gt, y_pred=pred)
        test_OP, test_PC, test_mIoU = cal_OP_PC_mIoU(test_cm)
        test_OPs.append(test_OP)
        test_PCs.append(test_PC)
        test_mIoUs.append(test_mIoU)

        e_time = time.time()

        log.info('[Epoch: %d/%d (%ds)] test acc: %.4f, OP: %.4f, PC: %.4f, mIoU: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time,
                  test_acc, test_OP, test_PC, test_mIoU))
        log.info(
            '-----------------------------------------------------------------------')

        if val_loss < best_validation_loss:
            torch.save(model.state_dict(), result_path + 'best_model.pth')

            save_confusion_matrix(cm=train_cm, path=result_path+'train_cm.png',
                                  title='epoch: %d, acc: %.4f, OP: %.4f, PC: %.4f, mIoU: %.4f' % (epoch+1, train_acc, train_OP, train_PC, train_mIoU))
            save_confusion_matrix(cm=test_cm, path=result_path+'test_cm.png',
                                  title='epoch: %d, acc: %.4f, OP: %.4f, PC: %.4f, mIoU: %.4f' % (epoch+1, test_acc, test_OP, test_PC, test_mIoU))

            best_validation_loss = val_loss
            final_OP = test_OP
            final_PC = test_PC
            final_mIoU = test_mIoU

        np.save(result_path+'train_OP', train_OPs)
        np.save(result_path+'train_PC', train_PCs)
        np.save(result_path+'train_mIoU', train_mIoUs)
        np.save(result_path+'test_OP', test_OPs)
        np.save(result_path+'test_PC', test_PCs)
        np.save(result_path+'test_mIoU', test_mIoUs)
        plt.plot(train_OPs, label='train_OP')
        plt.plot(train_PCs, label='train_PC')
        plt.plot(train_mIoUs, label='train_mIoU')
        plt.plot(test_OPs, label='test_OP')
        plt.plot(test_PCs, label='test_PC')
        plt.plot(test_mIoUs, label='test_mIoU')
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.savefig(result_path+'acc.png')
        plt.close()

        np.save(result_path+'train_loss', train_losses)
        np.save(result_path+'val_loss', val_losses)
        plt.plot(train_losses, label='train_loss')
        plt.plot(val_losses, label='val_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(result_path+'loss.png')
        plt.close()

    log.info(OmegaConf.to_yaml(cfg))
    log.info('OP: %.4f, PC: %.4f, mIoU: %.4f' %
             (final_OP, final_PC, final_mIoU))
    log.info('--------------------------------------------------')


if __name__ == '__main__':
    main()
