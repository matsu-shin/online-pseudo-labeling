from cProfile import label
from load_mnist import load_minist
import random
import numpy as np
import matplotlib.pyplot as plt


def get_label_proportion(num_bags=100, num_instances=100, num_classes=3):
    random.seed(0)

    label_proportion = np.zeros((num_bags, num_classes))
    remain_label_proportion = np.ones(num_bags)*100
    for n in range(num_instances):
        for c in range(num_classes-1):
            max = int(remain_label_proportion[n]+1)
            dice = np.arange(max)
            w = ((dice/max)-0.5)**2
            p = random.choices(dice, k=1, weights=w)[0]
            label_proportion[n][c] = p
            remain_label_proportion[n] -= p
    label_proportion[:, -1] = remain_label_proportion
    for n in label_proportion:
        random.shuffle(n)
    # label_proportion.shape => (num_bags, num_classes)
    return label_proportion/100


def create_bags(data, label, label_proportion, num_bags=1000, num_instances=100):
    # negative_data = data[label == negative_label]
    # positive_data = data[label == positive_label]

    # label_proportion = get_label_proportion(num_bags, num_instances)

    random.seed(0)
    bags_data, bags_label = [], []
    num_classes = label_proportion.shape[1]
    for n in range(num_bags):
        bag_data, bag_label = [], []
        for c in range(num_classes):
            data_c = data[label == c]
            if (c+1) != num_classes:
                num_c = int(num_instances*label_proportion[n][c])
            else:
                num_c = num_instances-len(bag_label)
            index = random.sample(list(range(len(data_c))), num_c)
            bag_data.extend(data_c[index])
            bag_label.extend([c]*num_c)

        bags_data.append(np.array(bag_data))
        bags_label.append(np.array(bag_label))

    bags_data, bags_label = np.array(bags_data), np.array(bags_label)
    # bags_data.shape => (num_bags, num_instances, (image_size))
    # bags_label.shape => (num_bags, num_instances)

    return np.array(bags_data), np.array(bags_label)


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = load_minist(
        dataset_dir='./data/MNIST/raw/',
        is_to_rgb=True)

    num_bags = 100
    num_instances = 100
    num_classes = 3
    label_proportion = get_label_proportion(
        num_bags=num_bags, num_instances=num_instances, num_classes=num_classes)
    for c in range(num_classes):
        plt.hist(label_proportion[:, c], alpha=0.5)
    plt.savefig('result/label_proportion.png')

    bags_data, bags_label = create_bags(
        train_data, train_label, label_proportion,
        num_bags=num_bags,
        num_instances=num_instances)
    print(bags_data.shape, bags_label.shape)

    # show a bag
    for index in range(3):
        fig, ax = plt.subplots(10, 10)
        for i in range(100):
            ax[i//10][i % 10].imshow(bags_data[index][i])
        fig.savefig('./result/bags_%s.png' % index)
