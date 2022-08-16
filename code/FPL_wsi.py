# Follow-the-Perturbed-Leader (FPL)
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from pulp import *
from mip import *


class FPL:
    def __init__(self, num_instances_dict, proportion_dict, sigma, eta, loss_f, pseudo_ratio, is_online_prediction=True):
        self.num_instances_dict = num_instances_dict
        self.proportion_dict = proportion_dict
        self.num_classes = len(self.proportion_dict[100])
        self.sigma = sigma
        self.eta = eta
        self.loss_f = loss_f
        self.pseudo_ratio = pseudo_ratio
        self.is_online_prediction = is_online_prediction

        self.k_dict = {}
        for idx, num_instances in num_instances_dict.items():
            label_proportion = np.array(self.proportion_dict[idx])
            n = num_instances * label_proportion * self.pseudo_ratio
            int_n = n.astype(int)
            for i in np.argsort(n-int_n)[::-1]:
                if int_n.sum() == int(num_instances*self.pseudo_ratio):
                    break
                int_n[i] += 1
            self.k_dict[idx] = int_n

        self.original_k_dict = {}
        for idx, num_instances in num_instances_dict.items():
            label_proportion = np.array(self.proportion_dict[idx])
            n = num_instances * label_proportion
            int_n = n.astype(int)
            for i in np.argsort(n-int_n)[::-1]:
                if int_n.sum() == num_instances:
                    break
                int_n[i] += 1
            self.original_k_dict[idx] = int_n

        random.seed(0)
        self.d_dict = {}

        self.theta = {}
        self.cumulative_loss = {}
        for idx, num_instances in num_instances_dict.items():
            self.cumulative_loss[idx] = np.zeros(
                (num_instances, self.num_classes))

    def update_theta(self, model, loader, num_instances_dict, device):
        model.eval()
        confidence_list = []

        for data in tqdm(loader, leave=False):
            data = data.to(device)
            y = model(data)
            confidence = F.softmax(y, dim=1)
            confidence_list.extend(confidence.cpu().detach().numpy())

        confidence = np.array(confidence_list)
        loss = 1-confidence
        x = 0
        for idx, num_instances in num_instances_dict.items():
            self.theta[idx] = loss[x: x+num_instances]
            x += num_instances

    # def update_d(self, num_instances_dict):
    #     for idx, num_instances in tqdm(num_instances_dict.items(), leave=False):
    #         perturbation = np.random.normal(
    #             0, self.sigma, (num_instances, self.num_classes))
    #         self.cumulative_loss[idx] += self.theta[idx]
    #         total_loss = self.cumulative_loss[idx] + self.eta*perturbation

    #         A = np.repeat(np.arange(num_instances), self.num_classes)
    #         A = np.identity(num_instances)[A]
    #         B = np.tile(np.arange(self.num_classes), num_instances)
    #         B = np.identity(self.num_classes)[B]

    #         total_loss = total_loss.reshape(-1)
    #         k = self.k_dict[idx]

    #         m = Model()
    #         d = m.add_var_tensor((num_instances*self.num_classes, ),
    #                              'd', var_type=BINARY)
    #         m.objective = minimize(
    #             xsum(x*y for x, y in zip(total_loss, d)))
    #         for i, a in enumerate(A.transpose()):
    #             m += xsum(x*y for x, y in zip(a[i*self.num_classes: (i+1)*self.num_classes],
    #                                           d[i*self.num_classes: (i+1)*self.num_classes])) <= 1
    #         for i, b in enumerate(B.transpose()):
    #             m += xsum(x*y for x, y in zip(b, d)) == k[i]

    #         m.verbose = 0
    #         # m.threads = -1
    #         m.optimize()
    #         d = np.array(d.astype(float)).reshape(
    #             num_instances, self.num_classes)

    #         self.d_dict[idx] = d.argmax(1)
    #         self.d_dict[idx][d.sum(axis=1) != 1] = -1

    def update_d(self, num_instances_dict):
        for idx, num_instances in tqdm(num_instances_dict.items(), leave=False):
            perturbation = np.random.normal(
                0, self.sigma, (num_instances, self.num_classes))
            self.cumulative_loss[idx] += self.theta[idx]
            total_loss = self.cumulative_loss[idx] + self.eta*perturbation

            A = np.repeat(np.arange(num_instances), self.num_classes)
            A = np.identity(num_instances)[A]
            B = np.tile(np.arange(self.num_classes), num_instances)
            B = np.identity(self.num_classes)[B]

            total_loss = total_loss.reshape(-1)
            k = self.original_k_dict[idx]

            m = Model()
            d = m.add_var_tensor((num_instances*self.num_classes, ),
                                 'd', var_type=BINARY)
            m.objective = minimize(
                xsum(x*y for x, y in zip(total_loss, d)))
            for i, a in enumerate(A.transpose()):
                m += xsum(x*y for x, y in zip(a[i*self.num_classes: (i+1)*self.num_classes],
                                              d[i*self.num_classes: (i+1)*self.num_classes])) <= 1
            for i, b in enumerate(B.transpose()):
                m += xsum(x*y for x, y in zip(b, d)) == k[i]

            m.verbose = 0
            # m.threads = -1
            m.optimize()
            d = np.array(d.astype(float)).reshape(
                num_instances, self.num_classes)
            self.d_dict[idx] = d.argmax(1)

            total_loss = total_loss.reshape(num_instances, self.num_classes)
            total_loss = total_loss[d.astype(bool)]
            for c in range(self.num_classes):
                c_index = np.where(self.d_dict[idx] == c)[0]
                c_loss = total_loss[self.d_dict[idx] == c]
                c_sorted_index = c_index[np.argsort(c_loss)]
                c_not_used_index = c_sorted_index[self.k_dict[idx][c]:]
                self.d_dict[idx][c_not_used_index] = -1


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
