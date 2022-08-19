# Follow-the-Perturbed-Leader (FPL)
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from mip import *


class FPL:
    def __init__(self, label_proportion, num_bags, num_instances, sigma, eta, loss_f, pseudo_ratio, is_online_prediction=True):
        self.label_proportion = label_proportion
        self.num_bags = num_bags
        self.num_instances = num_instances
        self.num_classes = label_proportion.shape[1]
        self.N = self.num_bags*self.num_instances
        self.sigma = sigma
        self.eta = eta
        self.loss_f = loss_f
        self.pseudo_ratio = pseudo_ratio
        self.is_online_prediction = is_online_prediction

        self.k = np.zeros((self.num_bags, self.num_classes), dtype=int)
        for n in range(num_bags):
            n_c = num_instances * label_proportion[n] * self.pseudo_ratio
            int_n_c = (n_c).astype(int)
            for idx in np.argsort(n_c-int_n_c)[::-1]:
                if int_n_c.sum() == int(self.num_instances*self.pseudo_ratio):
                    break
                int_n_c[idx] += 1
            self.k[n] = int_n_c
        # k.shape => (num_bags, num_classes)

        random.seed(0)
        self.d = np.zeros((self.num_bags, self.num_instances), dtype=int)
        for n in range(num_bags):
            d_n = []
            accum_n = 0
            for c in range(self.num_classes-1):
                n_c = int(num_instances * label_proportion[n][c])
                accum_n += n_c
                d_n.extend([c]*int(n_c))
            n_c = int(num_instances-accum_n)
            d_n.extend([self.num_classes-1]*int(n_c))

            random.shuffle(d_n)
            self.d[n] = d_n
        # d.shape = > (num_bags, num_instances)

        self.change_p_label_rate = []

        random.seed(0)
        self.select_idx = np.ones(
            (self.num_bags, self.num_instances), dtype=int)
        # select_idx.shape => (num_bags, num_instances)

        self.theta = np.zeros(
            (self.num_bags, self.num_instances, self.num_classes))
        self.cumulative_loss = np.zeros(
            (self.num_bags, self.num_instances, self.num_classes))

    def update_theta(self, model, criterion, loader, device):
        model.eval()
        confidence_list = []

        loss_list, acc_list = [], []
        pred_list, feature_list = [], []

        for data, label in loader:
            (_, _, w, h, c) = data.size()
            data = data.reshape(-1, w, h, c)
            label = label.reshape(-1)

            data, label = data.to(device), label.to(device)

            # model.avgpool.register_forward_hook(store_feature)
            y = model(data)
            confidence = F.softmax(y, dim=1)

            pred_list.extend(y.argmax(1).cpu().detach().numpy())
            confidence_list.extend(confidence.cpu().detach().numpy())
            # feature_list.extend(feature.squeeze().cpu().detach().numpy())

            loss_list.append(criterion(y, label).item())
            acc_list.extend((y.argmax(1) == label).cpu().detach().numpy())

        confidence = np.array(confidence_list)
        confidence = confidence.reshape(
            self.num_bags, self.num_instances, self.num_classes)
        # confidence.shape => (num_bags, num_instances, num_classes)

        if self.loss_f == 'song':
            self.theta = song_loss(confidence, self.d)
        elif self.loss_f == 'simple_confidence':
            self.theta = simple_confidence_loss(confidence)
        else:
            print('Error: No loss function!')

        loss = np.array(loss_list).mean()
        acc = np.array(acc_list).mean()
        pred = np.array(pred_list)

        return loss, acc, pred

    def update_d(self):
        d_tmp = self.d.copy()

        if self.is_online_prediction:
            perturbation = \
                np.random.normal(0, self.sigma, self.cumulative_loss.shape)

            self.cumulative_loss += self.theta
            total_loss = self.cumulative_loss + self.eta*perturbation
            # total_loss.shape => (num_bags, num_instances, num_classes)
        else:
            total_loss = self.theta

        A = np.repeat(np.arange(self.num_instances), self.num_classes)
        A = np.identity(self.num_instances)[A]
        B = np.tile(np.arange(self.num_classes), self.num_instances)
        B = np.identity(self.num_classes)[B]

        print('seletcing pseudo label...')
        for n in tqdm(range(self.num_bags)):
            total_loss_n = total_loss[n].reshape(-1)
            k_n = self.k[n]

            # mip
            m = Model()
            d_n = m.add_var_tensor((self.num_instances*self.num_classes, ),
                                   'd', var_type=BINARY)
            m.objective = minimize(
                xsum(x*y for x, y in zip(total_loss_n, d_n)))
            for i, a in enumerate(A.transpose()):
                m += xsum(x*y for x, y in zip(a[i*self.num_classes: (i+1)*self.num_classes],
                                              d_n[i*self.num_classes: (i+1)*self.num_classes])) <= 1
            for i, b in enumerate(B.transpose()):
                m += xsum(x*y for x, y in zip(b, d_n)) == k_n[i]

            m.verbose = 0
            # m.threads = -1
            m.optimize()
            d_n = np.array(d_n.astype(float)).reshape(
                self.num_instances, self.num_classes)

            self.d[n] = d_n.argmax(1)
            self.select_idx[n] = d_n.sum(axis=1)

        self.change_p_label_rate.append((self.d != d_tmp).mean())


def song_loss(confidence, label):
    # confidence.shape => (num_bags, num_instances, num_classes)
    # label => (num_bags, num_instances)
    (num_bags, num_instances, num_classes) = confidence.shape
    confidence = confidence.reshape(-1, num_classes)
    label = label.reshape(-1)

    correct = np.ones(confidence.shape)
    pred = confidence.argmax(1)
    pred_one_hot = np.identity(num_classes)[pred]
    label_one_hot = np.identity(num_classes)[label]
    correct[pred_one_hot != label_one_hot] = -1

    loss = ((1 - correct * confidence) / 2)
    loss = loss.reshape(num_bags, num_instances, num_classes)

    return loss


def simple_confidence_loss(confidence):
    return 1-confidence


def store_feature(module, input, output):
    global feature
    feature = output
