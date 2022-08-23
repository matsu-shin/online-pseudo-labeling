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

import torchvision.transforms as transforms
from utils import Dataset, fix_seed, get_rampup_weight, make_folder
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
        (self.n, self.b, self.w, self.h, self.c) = self.bags.shape
        self.len = self.n

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        index = np.arange(self.b)
        sampled_index = np.random.choice(index, self.num_sampled_instances)

        bag = torch.zeros((self.num_sampled_instances, self.c, self.w, self.h))
        for i, j in enumerate(sampled_index):
            bag[i] = self.transform(self.bags[idx][j])

        label_proportion = self.label_proportions[idx]
        label_proportion = torch.tensor(label_proportion).float()

        return bag, label_proportion


@ hydra.main(config_path='../config', config_name='config_toy_proportion_loss')
def main(cfg: DictConfig) -> None:

    # file name
    cwd = hydra.utils.get_original_cwd()
    result_path = cwd + cfg.result_path
    result_path += 'toy_proportion_loss/'
    make_folder(result_path)
    result_path += '%s' % str(cfg.dataset.name)
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
    num_bags = bags_index.shape[0]
    num_instances = bags_index.shape[1]

    bags_data, bags_label = train_data[bags_index], train_label[bags_index]
    label_proportion = np.zeros((num_bags, num_classes))
    for n in range(num_bags):
        bag_one_hot_label = np.identity(num_classes)[bags_label[n]]
        label_proportion[n] = bag_one_hot_label.sum(axis=0)/num_instances

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

    train_dataset = DatasetBagSampling(
        bags_data, label_proportion,
        num_sampled_instances=cfg.num_sampled_instances)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.mini_batch,
        shuffle=True,  num_workers=cfg.num_workers)

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

    test_acces = []
    for epoch in range(cfg.num_epochs):

        # train
        model.train()
        train_losses = []
        for iter, (data, proportion) in enumerate(tqdm(train_loader, leave=False)):
            data = data.to(cfg.device)
            proportion = proportion.to(cfg.device)

            (n, b, w, h, c) = data.size()
            data = data.reshape(-1, w, h, c)

            if cfg.consistency == "vat":
                # VAT should be calculated before the forward for cross entropy
                consistency_loss = consistency_criterion(model, data)
            elif cfg.consistency == "pi":
                consistency_loss, _ = consistency_criterion(model, data)
            else:
                consistency_loss = torch.tensor(0.)
            alpha = get_rampup_weight(0.05, iter, -1)
            consistency_loss = alpha * consistency_loss

            y = model(data)
            confidence = F.softmax(y, dim=1)
            confidence = confidence.reshape(n, b, -1).mean(dim=1)
            prop_loss = loss_function(confidence, proportion)

            loss = prop_loss + consistency_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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

        # print result
        log.info('[Epoch: %d/%d] train_loss: %.4f, test_acc: %.4f' %
                 (epoch+1, cfg.num_epochs, train_loss, test_acc))

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
