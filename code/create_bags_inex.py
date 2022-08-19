from load_mnist import load_minist
from load_cifar10 import load_cifar10
import random
import numpy as np
import matplotlib.pyplot as plt


def get_label_proportion(num_bags=100, num_instances=100, num_classes=10):
    random.seed(0)

    label_proportion = np.zeros((num_bags, num_classes))
    remain_label_proportion = np.ones(num_bags)*100
    for n in range(num_instances):
        for c in range(num_classes-1):
            max = int(remain_label_proportion[n]+1)
            dice = np.arange(max)
            # w = ((dice/max)-0.5)**2
            # # w = abs((dice/max)-0.5)
            # p = random.choices(dice, k=1, weights=w)[0]
            p = random.choices(dice, k=1)[0]
            label_proportion[n][c] = p
            remain_label_proportion[n] -= p
    label_proportion[:, -1] = remain_label_proportion
    for n in label_proportion:
        random.shuffle(n)
    # label_proportion.shape => (num_bags, num_classes)
    return label_proportion/100


def create_bags_index(label, label_proportion, num_bags=1000, num_instances=64):
    random.seed(0)
    num_classes = label_proportion.shape[1]

    bags_index = []
    for n in range(num_bags):
        bag_index = []
        for c in range(num_classes):
            c_index = np.where(label == c)[0]
            if (c+1) != num_classes:
                num_c = int(num_instances*label_proportion[n][c])
            else:
                num_c = num_instances-len(bag_index)

            sample_c_index = random.sample(list(c_index), num_c)
            bag_index.extend(sample_c_index)

        random.shuffle(bag_index)
        bags_index.append(bag_index)

    bags_index = np.array(bags_index)
    # bags_index.shape => (num_bags, num_instances)

    return bags_index


def create_uniform_bags_index(label, num_bags=1000, num_instances=64):
    random.seed(0)
    N = label.shape[0]

    bags_index = []
    for _ in range(num_bags):
        sample_index = random.sample(list(range(N)), num_instances)
        bags_index.append(sample_index)

    bags_index = np.array(bags_index)
    # bags_index.shape => (num_bags, num_instances)

    # return bags_index

    label_proportion = np.zeros((num_bags, 10))
    for n in range(num_bags):
        for l in label[bags_index[n]]:
            label_proportion[n][l] += 1
    label_proportion = label_proportion/64

    return bags_index, label_proportion


def create_bias_bags_index(label, num_bags=1000, num_instances=1024):
    np.random.seed(0)

    # make label proportion
    label_proportion = []
    lp = np.array([2, 2, 4, 8, 16, 32, 64, 128, 256, 512]) / 1024
    for i in range(num_bags):
        lp = np.random.permutation(lp)
        label_proportion.append(lp)
    label_proportion = np.array(label_proportion)

    bags_index = []
    for n in range(num_bags):
        bag_index = []
        for c in range(num_classes):
            c_index = np.where(label == c)[0]
            if (c+1) != num_classes:
                num_c = int(num_instances*label_proportion[n][c])
            else:
                num_c = num_instances-len(bag_index)

            sample_c_index = random.sample(list(c_index), num_c)
            bag_index.extend(sample_c_index)

        random.shuffle(bag_index)
        bags_index.append(bag_index)

    bags_index = np.array(bags_index)
    # bags_index.shape => (num_bags, num_instances)

    return bags_index, label_proportion


if __name__ == '__main__':
    num_bags = 500
    num_instances = 1024

    # train_data, train_label, _, _ = load_minist(
    #     dataset_dir='./data/MNIST/raw/')
    train_data, train_label, _, _ = load_cifar10(dataset_dir='./data/')
    num_classes = train_label.max()+1
    bags_index, label_proportion = create_bias_bags_index(
        label=train_label, num_bags=num_bags, num_instances=num_instances)
    print(bags_index.shape, label_proportion.shape)
    np.save('./obj/cifar10/bias-%d-index.npy' %
            (num_instances), bags_index)
    np.save('./obj/cifar10/bias-%d-proporton.npy' %
            (num_instances), label_proportion)

    # # num_instances = 100
    # for num_instances in [16, 32, 64, 128]:
    #     # train_data, train_label, _, _ = load_cifar10(dataset_dir='./data/')
    #     train_data, train_label, _, _ = load_minist(dataset_dir='./data/')
    #     num_classes = train_label.max()+1
    #     # num_classes = 3
    #     label_proportion = get_label_proportion(
    #         num_bags=num_bags, num_instances=num_instances, num_classes=num_classes)

    #     bags_index = create_bags_index(
    #         train_label, label_proportion, num_bags=num_bags, num_instances=num_instances)
    #     print(bags_index.shape)
    #     # np.save('./obj/cifar10/xx-%d.npy' % (num_instances), bags_index)
    #     np.save('./obj/cifar10/no_w-%d.npy' % (num_instances), bags_index)

    #     for c in range(num_classes):
    #         plt.hist(label_proportion[:, c], alpha=0.5)
    #     plt.ylim((0, 50))
    #     # plt.savefig('./result/debug/xx-%d.png' % (num_instances))
    #     plt.savefig('./result/debug/no_w-%d.png' % (num_instances))
    #     plt.close()

    # for num_instances in [16, 32, 64, 128]:
    #     train_data, train_label, _, _ = load_minist(
    #         dataset_dir='./data/MNIST/raw/')
    #     num_classes = train_label.max()+1
    #     bags_index, label_proportion = create_uniform_bags_index(
    #         train_label, num_bags=num_bags, num_instances=num_instances)
    #     print(bags_index.shape)
    #     np.save('./obj/mnist/uniform-SWOR-%d.npy' %
    #             (num_instances), bags_index)

    #     for c in range(num_classes):
    #         plt.hist(label_proportion[:, c], alpha=0.5)
    #     # plt.ylim((0, 50))
    #     # plt.savefig('./result/debug/xx-%d.png' % (num_instances))
    #     plt.savefig('./result/debug/mnist-uniform-%d.png' % (num_instances))
    #     plt.close()
