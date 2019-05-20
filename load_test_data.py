import numpy as np
import _pickle as cPickle




def load_test_data():
    with open('bin/data_other_test_image_ori_data_test_0', 'rb') as f:
        dic = cPickle.load(f)
        test_images = dic['data']
        test_images_size = len(test_images)
        test_images = np.reshape(test_images, (test_images_size, 80, 80, 3))
        test_labels = dic['labels']
    return (test_images, test_labels)

def load_single_test_data():
    with open('bin/data_1_face.jpg_ori_data_test_0', 'rb') as f:
        dic = cPickle.load(f)
        test_images = dic['data']
        test_images_size = len(test_images)
        test_images = np.reshape(test_images, (test_images_size, 80, 80, 3))
        test_labels = dic['labels']
    return (test_images, test_labels)