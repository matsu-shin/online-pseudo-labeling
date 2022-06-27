import random
import numpy as np
import torch.nn.functional as F


def pseudo_label_random(label_proportion, num_bags, num_instances):
    d = []
    for n in range(num_bags):
        num_positive = int(num_instances * label_proportion[n])
        num_negative = num_instances-num_positive
        d_n = np.concatenate(
            [np.zeros(num_negative), np.ones(num_positive)])
        random.shuffle(d_n)
        d.extend(d_n)
    d = np.array(d)

    return d


class PseudoLabelThreshould:
    def __init__(self, label_proportion, num_bags, num_instances):
        self.label_proportion = label_proportion
        self.num_bags = num_bags
        self.num_instances = num_instances
        self.N = self.num_bags*self.num_instances

        random.seed(0)
        d = []
        for n in range(num_bags):
            num_positive = int(num_instances * label_proportion[n])
            num_negative = num_instances-num_positive
            d_n = np.concatenate(
                [np.zeros(num_negative), np.ones(num_positive)])
            random.shuffle(d_n)
            d.extend(d_n)
        self.d = np.array(d)
        self.thera = np.zeros(self.N)

    def update_theta(self, model, loader, device):
        model.eval()
        theta = []
        for data, _ in loader:
            data = data.to(device)
            y = model(data)
            confidence = 1 - F.softmax(y, dim=1).cpu().detach().numpy()
            theta.extend(confidence[:, 1])
        self.theta = np.array(theta)

    def update_d(self):
        # threshould
        theta = self.theta.reshape(self.num_bags, self.num_instances)
        d = []
        for n in range(self.num_bags):
            k = int(self.num_instances * self.label_proportion[n])
            loss = theta[n]
            d_n = np.zeros(self.num_instances)
            for i in np.argsort(loss)[:k]:
                d_n[i] = 1
            d.extend(d_n)

        self.d = np.array(d)
