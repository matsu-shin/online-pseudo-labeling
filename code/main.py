import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from tqdm import tqdm

from FPL import FPL
from evaluate import evaluate
from train import train
from utils import Dataset
from create_bags import create_bags, get_label_proportion
from load_mnist import load_minist
from load_cifar10 import load_cifar10

log = logging.getLogger(__name__)


@ hydra.main(config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:

    log.info(OmegaConf.to_yaml(cfg))
    cwd = hydra.utils.get_original_cwd()
    log.info('cwd:%s' % cwd)

    num_bags = cfg.dataset.num_bags
    num_instances = cfg.dataset.num_instances
    num_classes = cfg.dataset.num_classes

    # load data
    if cfg.dataset.name == 'MNIST':
        train_data, train_label, test_data, test_label = \
            load_minist(
                dataset_dir=cwd+cfg.dataset.dir,
                is_to_rgb=cfg.dataset.is_to_rgb)
    if cfg.dataset.name == 'CIFAR10':
        train_data, train_label, test_data, test_label = \
            load_cifar10(
                dataset_dir=cwd+cfg.dataset.dir
            )

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

    # create bugs
    label_proportion = get_label_proportion(
        num_bags=num_bags, num_instances=num_instances, num_classes=num_classes)
    # label_proportion.shape => (num_bags, num_classes)

    bags_data, bags_label = create_bags(
        train_data, train_label, label_proportion,
        num_bags=num_bags,
        num_instances=num_instances)
    # bags_data.shape => (num_bags, num_instances, (image_size))
    # bags_label.shape => (num_bags, num_instances)

    N = cfg.dataset.img_size
    train_data = bags_data.reshape(-1, N, N, cfg.dataset.img_ch)
    train_label = bags_label.reshape(-1)
    # bags_data.shape => (num_bags*num_instances, (image_size))
    # bags_label.shape => (num_bags*num_instances)

    # define model
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
              is_online_prediction=cfg.fpl.is_online_prediction)

    test_loss_list, test_acc_list = [], []
    for epoch in range(cfg.num_epochs):
        # update pseudo label
        pseudo_label = fpl.d
        # pseudo_label.shape => (num_bags, num_instances)

        # trained DNN: obtain f_t with d_t
        train_dataset = Dataset(train_data, pseudo_label.reshape(-1))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            shuffle=True,  num_workers=cfg.num_workers)

        train_loss, train_acc = train(model=model,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      loader=train_loader,
                                      device=cfg.device)
        # print(train_loss, train_acc)

        # update the noise-risk vector theta_t
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            shuffle=False,  num_workers=cfg.num_workers)
        fpl.update_theta(model=model,
                         loader=train_loader,
                         device=cfg.device)

        # select the next k-set d_(t+1)
        fpl.update_d()

        test_loss, test_acc = evaluate(model=model,
                                       criterion=criterion,
                                       loader=test_loader,
                                       device=cfg.device)

        # print result
        print('[Epoch: %d] test_loss: %.4f, test_acc: %.4f' %
              (epoch+1, test_loss, test_acc))
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

    # save
    if cfg.fpl.is_online_prediction:
        f_name = 'fpl'
    else:
        f_name = 'not_op'
    f_name += '_%s_%s_%s_%s_%s_%s' % (
        cfg.dataset.name,
        cfg.dataset.num_classes,
        cfg.fpl.eta,
        cfg.fpl.loss_f,
        cfg.dataset.num_bags,
        cfg.dataset.num_instances)
    np.save('% s/result/loss_%s' % (cwd, f_name), np.array(test_loss_list))
    np.save('% s/result/acc_%s' % (cwd, f_name), np.array(test_acc_list))


if __name__ == '__main__':
    main()
