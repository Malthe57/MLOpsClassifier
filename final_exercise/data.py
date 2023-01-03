import torch
import numpy as np
import os

def mnist():
    train = np.load("../../../data/corruptmnist/train_{}.npz".format(0))
    images_vec = train['images']
    labels_vec = train['labels']
    for i in range(1,5):
        train = np.load("../../../data/corruptmnist/train_{}.npz".format(i))
        images = train['images']
        labels = train['labels']
        images_vec = np.concatenate((images_vec, images))
        labels_vec = np.concatenate((labels_vec, labels))
    train = {'images': images_vec, 'labels':labels_vec}
    test = np.load("../../../data/corruptmnist/test.npz")
    test = {'images':test['images'],'labels':test['labels']}
    return train, test

train_set = mnist()
print('os.path')