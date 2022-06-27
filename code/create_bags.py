from load_mnist import load_minist
import random
import numpy as np
import matplotlib.pyplot as plt


def get_label_proportion(num_bags=1000, num_instances=100):
    random.seed(0)
    dice = list(range(num_instances+1))
    w = [x**2 for x in np.arange(-1, 1, 2/(num_instances+1))]
    # w = [1-abs(x) for x in range(-1, 1, num_bags)]
    num_negative_list = random.choices(dice, k=num_bags, weights=w)

    # num_negative_list = random.choices(dice, k=num_bags)
    # num_negative_list = [50]*num_bags

    label_proportion = np.array(num_negative_list)/num_instances

    return label_proportion


def create_bags(data, label, num_bags=1000, num_instances=100, negative_label=0, positive_label=1):
    negative_data = data[label == negative_label]
    positive_data = data[label == positive_label]

    label_proportion = get_label_proportion(num_bags, num_instances)

    random.seed(0)
    bags_data, bags_label = [], []
    for lp in label_proportion:
        num_positive = int(num_instances * lp)
        num_negative = num_instances-num_positive
        negative_index = \
            random.sample(list(range(len(negative_data))), num_negative)
        positive_index = \
            random.sample(list(range(len(positive_data))), num_positive)
        bags_data.append(np.concatenate(
            [negative_data[negative_index], positive_data[positive_index]]))
        bags_label.append(np.concatenate(
            [np.zeros(num_negative), np.ones(num_positive)]))

    return np.array(bags_data), np.array(bags_label), label_proportion


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = load_minist(
        dataset_dir='./data/MNIST/raw/',
        is_to_rgb=True)

    num_bags = 1000
    num_instances = 100
    label_proportion = get_label_proportion()
    plt.hist(label_proportion, bins=num_instances+1)
    plt.savefig('./result/hist_label_proportion.png')

    bags_data, bags_label, label_proportion = create_bags(
        train_data, train_label)
    print(bags_data.shape, bags_label.shape)

    # show a bag
    index = 0
    fig, ax = plt.subplots(10, 10)
    for i in range(100):
        ax[i//10][i % 10].imshow(bags_data[index][i])
    fig.suptitle('lable ratio: %.2f' % label_proportion[index])
    fig.savefig('./result/bags_%s.png' % index)
