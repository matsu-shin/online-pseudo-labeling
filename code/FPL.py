# Follow-the-Perturbed-Leader (FPL)
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


class FPL:
    def __init__(self, label_proportion, num_bags, num_instances, sigma, eta):
        self.label_proportion = label_proportion
        self.num_bags = num_bags
        self.num_instances = num_instances
        self.N = self.num_bags*self.num_instances
        self.sigma = sigma
        self.eta = eta

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
        self.cumulative_loss = np.zeros(self.N)

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
        # TODO: 末廣先生に合っているか確認
        perturbation = np.random.normal(0, self.sigma, (self.N))
        self.cumulative_loss += self.d * self.theta
        total_loss = self.cumulative_loss + self.eta*self.d*perturbation

        total_loss = total_loss.reshape(self.num_bags, self.num_instances)
        d = []
        for n in range(self.num_bags):
            k = int(self.num_instances * self.label_proportion[n])
            loss = total_loss[n]
            d_n = np.zeros(self.num_instances)
            for i in np.argsort(loss)[:k]:
                d_n[i] = 1
            d.extend(d_n)

        self.d = np.array(d)
