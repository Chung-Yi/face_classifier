import numpy as np
import _pickle as cPickle


def load_data():
    with open('bin/data_train_ori_data_train_0', 'rb') as f:
        dic = cPickle.load(f)
        train_images = dic['data']
        train_images_size = len(train_images)
        train_images = np.reshape(train_images, (train_images_size, 80, 80, 3))
        train_labels = dic['labels']
    with open('bin/data_test_ori_data_test_0', 'rb') as f:
        dic = cPickle.load(f)
        test_images = dic['data']
        test_images_size = len(test_images)
        test_images = np.reshape(test_images, (test_images_size, 80, 80, 3))
        test_labels = dic['labels']
    return (train_images, train_labels), (test_images, test_labels)