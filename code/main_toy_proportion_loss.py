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
    def __init__(self, bags, labels, label_proportions, num_sampled_instances):
        self.bags = bags
        self.labels = labels
        self.label_proportions = label_proportions
        self.num_classes = (self.labels).max()+1
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
        label = torch.zeros(self.num_sampled_instances)
        for i, j in enumerate(sampled_index):
            bag[i] = self.transform(self.bags[idx][j])
            label[i] = self.labels[idx][j]

        label_proportion = self.label_proportions[idx]
        label_proportion = torch.tensor(label_proportion).float()

        return bag, label, label_proportion


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

    if cfg.validation>0:
        n_val = int(bags_index.shape[0]*cfg.validation)
        val_bags_index = bags_index[: n_val]
        bags_index = bags_index[n_val: ]

        bags_data, bags_label = train_data[val_bags_index], train_label[val_bags_index]
        label_proportion = np.identity(num_classes)[bags_label].mean(axis=1)
        val_dataset = DatasetBagSampling(
            bags_data, bags_label, label_proportion,
            num_sampled_instances=cfg.num_sampled_instances)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.mini_batch,
            shuffle=False,  num_workers=cfg.num_workers)

    # for train
    bags_data, bags_label = train_data[bags_index], train_label[bags_index]
    label_proportion = np.identity(num_classes)[bags_label].mean(axis=1)
    train_dataset = DatasetBagSampling(
        bags_data, bags_label, label_proportion,
        num_sampled_instances=cfg.num_sampled_instances)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.mini_batch,
        shuffle=True,  num_workers=cfg.num_workers)

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

    train_acces, val_acces, test_acces = [], [], []
    train_losses, val_losses = [], []
    best_validation_loss = np.inf
    final_acc = 0
    for epoch in range(cfg.num_epochs):
        # train
        model.train()
        losses = []
        gt, pred = [], []
        for iteration, (data, label, proportion) in enumerate(tqdm(train_loader, leave=False)):
            data, label = data.to(cfg.device), label.to(cfg.device)
            proportion = proportion.to(cfg.device)

            (n, b, w, h, c) = data.size()
            data = data.reshape(-1, w, h, c)
            label = label.reshape(-1)

            if cfg.consistency == "vat":
                # VAT should be calculated before the forward for cross entropy
                consistency_loss = consistency_criterion(model, data)
            elif cfg.consistency == "pi":
                consistency_loss, _ = consistency_criterion(model, data)
            else:
                consistency_loss = torch.tensor(0.)
            alpha = get_rampup_weight(0.05, iteration, -1)
            consistency_loss = alpha * consistency_loss

            y = model(data)
            confidence = F.softmax(y, dim=1)
            confidence = confidence.reshape(n, b, -1).mean(dim=1)
            prop_loss = loss_function(confidence, proportion)
            loss = prop_loss + consistency_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            gt.extend(label.cpu().detach().numpy())
            pred.extend(y.argmax(1).cpu().detach().numpy())
            losses.append(loss.item())

        train_loss = np.array(losses).mean()
        train_acc = (np.array(gt)==np.array(pred)).mean()
        train_losses.append(train_loss)
        train_acces.append(train_acc)
        log.info('[Epoch: %d/%d] train_loss: %.4f, train_acc: %.4f' %(epoch+1, cfg.num_epochs, train_loss, train_acc))

        # validation
        if cfg.validation>0:
            model.eval()
            losses = []
            gt, pred = [], []
            for data, label, proportion in tqdm(val_loader, leave=False):
                data, label = data.to(cfg.device), label.to(cfg.device)
                proportion = proportion.to(cfg.device)

                (n, b, w, h, c) = data.size()
                data = data.reshape(-1, w, h, c)
                label = label.reshape(-1)

                y = model(data)
                confidence = F.softmax(y, dim=1)
                confidence = confidence.reshape(n, b, -1).mean(dim=1)
                loss = loss_function(confidence, proportion)

                gt.extend(label.cpu().detach().numpy())
                pred.extend(y.argmax(1).cpu().detach().numpy())
                losses.append(loss.item())

            val_loss = np.array(losses).mean()
            val_acc = (np.array(gt)==np.array(pred)).mean()
            val_losses.append(val_loss)
            val_acces.append(val_acc)
            log.info('[Epoch: %d/%d] val_loss: %.4f, val_acc: %.4f' %(epoch+1, cfg.num_epochs, val_loss, val_acc))

        # test
        model.eval()
        gt, pred = [], []
        for data, label in tqdm(test_loader, leave=False):
            data, label = data.to(cfg.device), label.to(cfg.device)
            y = model(data)
            gt.extend(label.cpu().detach().numpy())
            pred.extend(y.argmax(1).cpu().detach().numpy())
        test_acc = (np.array(gt)==np.array(pred)).mean()
        test_acces.append(test_acc)

        log.info('[Epoch: %d/%d] test_acc: %.4f' %(epoch+1, cfg.num_epochs, test_acc))
        log.info('--------------------------------------------------')

        if cfg.validation>0:
            if val_loss < best_validation_loss:
                torch.save(model.state_dict(), result_path + 'best_model.pth')
                best_validation_loss = val_loss
                final_acc = test_acc

        np.save(result_path+'train_acc', train_acces)
        np.save(result_path+'val_acc', val_acces)
        np.save(result_path+'test_acc', test_acces)
        plt.plot(train_acces, label='train_acc')
        plt.plot(val_acces, label='val_acc')
        plt.plot(test_acces, label='test_acc')
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
    log.info('final_acc: %.4f' %(final_acc))
    log.info('--------------------------------------------------')


if __name__ == '__main__':
    main()
