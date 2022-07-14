import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from tqdm import tqdm
from matplotlib import pyplot as plt

from FPL import FPL
from train import TrainDataset, train
from evaluate import TestDataset, evaluate
from utils import make_folder, save_confusion_matrix
from create_bags import create_bags, get_label_proportion
from load_mnist import load_minist
from load_cifar10 import load_cifar10

log = logging.getLogger(__name__)


@ hydra.main(config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:

    log.info(OmegaConf.to_yaml(cfg))
    cwd = hydra.utils.get_original_cwd()
    log.info('cwd:%s' % cwd)

    # file name
    result_path = cfg.result_path
    if cfg.fpl.is_online_prediction:
        result_path += 'fpl'
    else:
        result_path += 'not_op'
    result_path += '_%s_%s_%s_%s_%s_%s_%s' % (
        cfg.dataset.name,
        cfg.is_uni,
        cfg.dataset.num_classes,
        cfg.fpl.eta,
        cfg.fpl.loss_f,
        cfg.dataset.num_bags,
        cfg.dataset.num_instances)

    # make folder
    make_folder(cwd+result_path)
    make_folder(cwd+result_path+'/model/')
    make_folder(cwd+result_path+'/train_cm/')
    make_folder(cwd+result_path+'/test_cm/')
    make_folder(cwd+result_path+'/label_cm/')
    make_folder(cwd+result_path+'/train_pseudo_cm/')
    make_folder(cwd+result_path+'/train_feature/')
    make_folder(cwd+result_path+'/test_feature/')
    make_folder(cwd+result_path+'/theta/')

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

    if cfg.is_load_bag:
        if cfg.is_uni == 1:
            dataset_path = cwd + \
                '/obj/%s/uniform-SWOR-%s.npy' % (cfg.dataset.name,
                                                 cfg.dataset.num_instances)
            print('loading '+dataset_path)
            bags_index = np.load(dataset_path)
        else:
            dataset_path = cwd + \
                '/obj/%s/xx-%s.npy' % (cfg.dataset.name,
                                       cfg.dataset.num_instances)
            print('loading '+dataset_path)
            bags_index = np.load(dataset_path)
        cfg.dataset.num_classes = 10
        cfg.dataset.num_bags = bags_index.shape[0]
        num_classes = cfg.dataset.num_classes
        num_instances = cfg.dataset.num_instances
        num_bags = cfg.dataset.num_bags

        bags_data, bags_label = train_data[bags_index], train_label[bags_index]
        label_proportion = np.zeros((num_bags, num_classes))
        for n in range(cfg.dataset.num_bags):
            bag_one_hot_label = np.identity(num_classes)[bags_label[n]]
            label_proportion[n] = bag_one_hot_label.sum(axis=0)/num_instances
    else:
        num_classes = cfg.dataset.num_classes
        num_instances = cfg.dataset.num_instances
        num_bags = cfg.dataset.num_bags
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
    # train_data.shape => (num_bags*num_instances, (image_size))
    # train_label.shape => (num_bags*num_instances)

    # for test
    # data, label = [], []
    # for c in range(num_classes):
    #     data.extend(test_data[test_label == c])
    #     label.extend(test_label[test_label == c])
    # test_data, test_label = np.array(data), np.array(label)
    # print(test_data.shape, test_label.shape)
    test_dataset = TestDataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False,  num_workers=cfg.num_workers)

    # define model
    model = resnet18(pretrained=cfg.is_pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(cfg.device)
    if cfg.is_pretrained_by_proportion_loss:
        # model_path = cwd+'/result/0701_class10_compare/model_uni_minib100_LLPVAT_propcons_cifar10_10_500_64.tar'
        model_path = cwd+'/result/pretrained_model.tar'
        model.load_state_dict(torch.load(model_path)['state_dict'])

    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    fpl = FPL(label_proportion=label_proportion,
              num_bags=num_bags, num_instances=num_instances,
              sigma=cfg.fpl.sigma,
              eta=cfg.fpl.eta,
              loss_f=cfg.fpl.loss_f,
              is_online_prediction=cfg.fpl.is_online_prediction)

    if cfg.is_pretrained_by_proportion_loss == 1:
        # test
        test_loss, test_acc, test_pred, feature = evaluate(model=model,
                                                           criterion=criterion,
                                                           loader=test_loader,
                                                           device=cfg.device)
        np.save('%s/%s/test_feature/0' % (cwd, result_path), feature)
        # print result
        print('[Epoch: 0/%d] test_loss: %.4f, test_acc: %.4f' %
              (cfg.num_epochs, test_loss, test_acc))
        save_confusion_matrix(
            test_label, test_pred,
            path='%s/%s/test_cm/0.png' % (cwd, result_path),
            epoch=0)

        # update the noise-risk vector theta_t
        train_dataset = TrainDataset(
            train_data, train_label, fpl.d.reshape(-1))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            shuffle=False,  num_workers=cfg.num_workers)
        train_loss, pseudo_loss, train_acc, pseudo_acc, train_pred, feature = fpl.update_theta(model=model,
                                                                                               criterion=criterion,
                                                                                               loader=train_loader,
                                                                                               device=cfg.device)
        np.save('%s/%s/train_feature/0' % (cwd, result_path), feature)
        np.save('%s/%s/theta/0' % (cwd, result_path), fpl.theta)

        print('[Epoch: 0/%d] train_loss: %.4f, train_acc: %.4f' %
              (cfg.num_epochs, train_loss, train_acc))
        print('[Epoch: 0/%d] train_pseudo_loss: %.4f, train_pseudo_acc: %.4f' %
              (cfg.num_epochs, pseudo_loss, pseudo_acc))

        # select the next k-set d_(t+1)
        fpl.update_d()
        pseudo_label_acc = (train_label == fpl.d.reshape(-1)).mean()
        print('pseudo_label_acc: %.4f' % pseudo_label_acc)
        save_confusion_matrix(
            train_label, fpl.d.reshape(-1),
            path='%s/%s/pseudo_cm/0.png' % (cwd, result_path),
            epoch=0)

    train_loss_list, train_acc_list = [], []
    train_pseudo_loss_list, train_pseudo_acc_list = [], []
    pseudo_label_acc_list = []
    test_loss_list, test_acc_list = [], []
    for epoch in range(cfg.num_epochs):
        # # update pseudo label
        # pseudo_label = fpl.d
        # # pseudo_label.shape => (num_bags, num_instances)

        # trained DNN: obtain f_t with d_t
        train_dataset = TrainDataset(
            train_data, train_label, fpl.d.reshape(-1))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            shuffle=True,  num_workers=cfg.num_workers)

        train(model=model,
              criterion=criterion,
              optimizer=optimizer,
              loader=train_loader,
              device=cfg.device)

        # update the noise-risk vector theta_t
        train_dataset = TrainDataset(
            train_data, train_label, fpl.d.reshape(-1))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            shuffle=False,  num_workers=cfg.num_workers)
        train_loss, pseudo_loss, train_acc, pseudo_acc, train_pred, feature = fpl.update_theta(model=model,
                                                                                               criterion=criterion,
                                                                                               loader=train_loader,
                                                                                               device=cfg.device)
        train_loss_list.append(train_loss)
        train_pseudo_loss_list.append(pseudo_loss)
        train_acc_list.append(train_acc)
        train_pseudo_acc_list.append(pseudo_acc)
        np.save('%s/%s/train_feature/%d' %
                (cwd, result_path, (epoch+1)), feature)
        np.save('%s/%s/theta/%d' %
                (cwd, result_path, (epoch+1)), fpl.theta)
        save_confusion_matrix(
            train_label, train_pred,
            path='%s/%s/train_cm/%d.png' % (cwd, result_path, epoch+1),
            epoch=(epoch+1))
        save_confusion_matrix(
            fpl.d.reshape(-1), train_pred,
            path='%s/%s/train_pseudo_cm/%d.png' % (cwd, result_path, epoch+1),
            epoch=(epoch+1))

        print('[Epoch: %d/%d] train_loss: %.4f, train_acc: %.4f' %
              (epoch+1, cfg.num_epochs, train_loss, train_acc))
        print('[Epoch: %d/%d] train_pseudo_loss: %.4f, train_pseudo_acc: %.4f' %
              (epoch+1, cfg.num_epochs, pseudo_loss, pseudo_acc))

        # select the next k-set d_(t+1)
        fpl.update_d()
        pseudo_label_acc = (train_label == fpl.d.reshape(-1)).mean()
        pseudo_label_acc_list.append(pseudo_label_acc)
        print('pseudo_label_acc: %.4f' % pseudo_label_acc)
        save_confusion_matrix(
            train_label, fpl.d.reshape(-1),
            path='%s/%s/label_cm/%d.png' % (cwd, result_path, epoch+1),
            epoch=(epoch+1))

        test_loss, test_acc, test_pred, feature = evaluate(model=model,
                                                           criterion=criterion,
                                                           loader=test_loader,
                                                           device=cfg.device)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        np.save('%s/%s/test_feature/%d' %
                (cwd, result_path, (epoch+1)), feature)
        save_confusion_matrix(
            test_label, test_pred,
            path='%s/%s/test_cm/%d.png' % (cwd, result_path, epoch+1),
            epoch=(epoch+1))

        # print result
        print('[Epoch: %d/%d] test_loss: %.4f, test_acc: %.4f' %
              (epoch+1, cfg.num_epochs, test_loss, test_acc))

        # save
        np.save('%s/%s/train_loss' %
                (cwd, result_path), np.array(train_loss_list))
        np.save('%s/%s/train_pseudo_loss' %
                (cwd, result_path), np.array(train_pseudo_loss_list))
        np.save('%s/%s/train_acc' %
                (cwd, result_path), np.array(train_acc_list))
        np.save('%s/%s/train_pseudo_acc' %
                (cwd, result_path), np.array(train_pseudo_acc_list))

        np.save('%s/%s/test_loss' %
                (cwd, result_path), np.array(test_loss_list))
        np.save('%s/%s/test_acc' %
                (cwd, result_path), np.array(test_acc_list))
        np.save('%s/%s/label_acc' %
                (cwd, result_path), np.array(pseudo_label_acc_list))

        torch.save(model.state_dict(), '%s/%s/model/%s.pth' %
                   (cwd, result_path, (epoch+1)))

        plt.plot(train_pseudo_acc_list, label='train_pseudo_acc')
        plt.plot(train_acc_list, label='train_acc')
        plt.plot(test_acc_list, label='test_acc')
        plt.plot(pseudo_label_acc_list, label='label_acc')
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.savefig('%s/%s/acc.png' % (cwd, result_path))
        plt.close()


if __name__ == '__main__':
    main()
