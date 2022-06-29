# Follow-the-Perturbed-Leader (FPL)
import random
from matplotlib.pyplot import axes
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from pulp import *


class FPL:
    def __init__(self, label_proportion, num_bags, num_instances, sigma, eta, loss_f, is_online_prediction=True):
        self.label_proportion = label_proportion
        self.num_bags = num_bags
        self.num_instances = num_instances
        self.num_classes = label_proportion.shape[1]
        self.N = self.num_bags*self.num_instances
        self.sigma = sigma
        self.eta = eta
        self.loss_f = loss_f
        self.is_online_prediction = is_online_prediction

        self.k = np.zeros((self.num_bags, self.num_classes), dtype=int)
        for n in range(num_bags):
            accum_n = 0
            for c in range(self.num_classes-1):
                n_c = int(num_instances * label_proportion[n][c])
                accum_n += n_c
                self.k[n][c] = n_c
            n_c = int(num_instances-accum_n)
            self.k[n][self.num_classes-1] = n_c
        # k.shape => (num_bags, num_classes)

        random.seed(0)
        self.d = np.zeros((self.num_bags, self.num_instances), dtype=int)
        for n in range(num_bags):
            d_n = []
            for c in range(self.num_classes):
                d_n.extend([c]*int(self.k[n][c]))
            random.shuffle(d_n)
            self.d[n] = d_n
        # d.shape => (num_bags, num_instances)

        self.theta = np.zeros(
            (self.num_bags, self.num_instances, self.num_classes))
        self.cumulative_loss = np.zeros(
            (self.num_bags, self.num_instances, self.num_classes))

    def update_theta(self, model, loader, device):
        model.eval()
        confidence_list = []
        for data, _ in loader:
            data = data.to(device)
            y = model(data)
            confidence = F.softmax(y, dim=1).cpu().detach().numpy()
            confidence_list.extend(confidence)
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

    def update_d(self):
        # TODO: 末廣先生に合っているか確認

        # to one-hot
        self.d_one_hot = np.identity(self.num_classes)[self.d]
        # self.d.shape => (num_bags, num_instances, num_classes)

        if self.is_online_prediction:
            perturbation = \
                np.random.normal(0, self.sigma, self.cumulative_loss.shape)

            self.cumulative_loss += self.d_one_hot * self.theta
            total_loss = self.cumulative_loss + self.eta*self.d_one_hot*perturbation
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

            m = LpProblem()
            d_n = [LpVariable('d%d' % i, cat=LpBinary) for i in range(
                self.num_instances*self.num_classes)]  # 変数
            m += lpDot(total_loss_n, d_n)  # objective funcation
            for a in A.transpose():
                m += lpDot(a, d_n) <= 1
            for i, b in enumerate(B.transpose()):
                m += lpDot(b, d_n) == k_n[i]
            m.solve(PULP_CBC_CMD(msg=False))
            d_n = [int(value(d_n[i]))
                   for i in range(self.num_instances*self.num_classes)]
            d_n = np.array(d_n).reshape(self.num_instances, self.num_classes)
            self.d[n] = d_n.argmax(1)


# def song_loss_original(confidence, label):
#     # confidence.shape => (num_bags*num_instances, num_classes)
#     # label => (num_bags*num_instances)
#     correct = np.ones(label.shape[0])
#     pred = confidence.argmax(1)
#     correct[pred != label] = -1
#     probality = confidence.max(1)

#     loss = ((1 - correct * probality) / 2)

#     return loss


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
