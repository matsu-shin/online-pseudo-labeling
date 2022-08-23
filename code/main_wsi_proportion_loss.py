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
    if cfg.consistency == 'none':
        consistency_criterion = None
    elif cfg.consistency == 'vat':
        consistency_criterion = VATLoss()
    elif cfg.consistency == 'pi':
        consistency_criterion = PiModelLoss()
    else:
        raise NameError('Unknown consistency criterion')

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

        b_list = [0]
        mb_data, mb_proportion = [], []
        for iter, (data, proportion) in enumerate(tqdm(train_loader, leave=False)):
            data = data[0]
            b_list.append(b_list[-1]+data.size(0))
            mb_data.extend(data)
            mb_proportion.extend(proportion)

            if (iter+1)%cfg.mini_batch==0 or (iter + 1)==len(train_loader):
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
                alpha = get_rampup_weight(0.05, iter, -1)
                consistency_loss = alpha * consistency_loss

                y = model(mb_data)
                confidence = F.softmax(y, dim=1)
                pred = torch.zeros(mb_proportion.size(0), cfg.dataset.num_classes).to(cfg.device)
                for n in range(mb_proportion.size(0)):
                    pred[n] = torch.mean(confidence[b_list[n]: b_list[n+1]], dim=0)
                prop_loss = loss_function(pred, mb_proportion)

                loss = prop_loss + consistency_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                b_list = [0]
                mb_data, mb_proportion = [], []

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

        test_cm = confusion_matrix(y_true=test_gt, y_pred=test_pred)
        test_OP, test_PC, test_mIoU = cal_OP_PC_mIoU(test_cm)
        save_confusion_matrix(cm=test_cm, path=result_path+'test_cm.png',
                              title='acc: %.4f, OP: %.4f, PC: %.4f, mIoU: %.4f' % (test_acc, test_OP, test_PC, test_mIoU))

        log.info('[Epoch: %d/%d (%ds)] train loss: %.4f, test acc: %.4f, OP: %.4f, PC: %.4f, mIoU: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time,
                  train_loss, test_acc, test_OP, test_PC, test_mIoU))

        np.save(result_path+'test_acc', test_acces)
        torch.save(model.state_dict(), result_path +
                   'model_'+str(epoch+1)+'.pth')
        plt.plot(test_acces, label='test_acc')
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.savefig(result_path+'acc.png')
        plt.close()

        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), result_path +
                       'model_'+str(epoch+1)+'.pth')


if __name__ == '__main__':
    main()
