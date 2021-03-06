import cv2
import os
import numpy as np
import face_recognition as fr
import random
import glob

SHAPE = 80
CHANNEL_LEN = SHAPE * SHAPE
DATA_LEN = SHAPE * SHAPE * 3


def imread(im_path, shape=None, mode=cv2.IMREAD_UNCHANGED):
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if shape != None:
        assert isinstance(shape, int)
        im = cv2.resize(im, (shape, shape))
    return im


def imagelist(data_path, file_list):
    with open(file_list, 'a') as f:
        if os.path.isdir(data_path):
            for filename in os.listdir(data_path):
                class_label = filename.split('_')[0]
                if filename != '.DS_Store':
                    f.write(filename + ' ' + class_label + '\n')
        elif os.path.isfile(data_path):
            class_label = data_path.split('_')[0]
            if data_path != '.DS_Store':
                f.write(data_path + ' ' + class_label)


def read_data(filename, data_path, shape=None):
    if os.path.isdir(filename):
        print("Can't found data file!")
    else:
        f = open(filename)
        lines = f.read().splitlines()

        lst = [ln.split(' ')[0] for ln in lines]
        label = [int(ln.split(' ')[1]) for ln in lines]
        face_image = []
        idx = 0
        s, c = SHAPE, CHANNEL_LEN
        for ln in lines:
            fname, lab = ln.split(' ')
            if fname == data_path:
                im = imread(fname, shape=s)
            else:
                im = imread(os.path.join(data_path, fname), shape=s)

            label[idx] = int(lab)
            face_image.append(im)
            idx += 1

        label = np.array(label, dtype=np.uint8)
        label = np.reshape(label, (len(label), 1))
        face_image = np.array(face_image)

        return face_image, label, lst


def face_encoding_read(filename, data_path, shape=None):
    if os.path.isdir(filename):
        print("Can't found data file!")
    else:
        f = open(filename)
        lines = f.read().splitlines()
        count = len(lines)
        data = np.zeros((count, 128))

        lst = [ln.split(' ')[0] for ln in lines]
        label = [int(ln.split(' ')[1]) for ln in lines]
        idx = 0
        s, c = SHAPE, CHANNEL_LEN

        for ln in lines:
            fname, lab = ln.split(' ')
            frame = fr.load_image_file(os.path.join(data_path, fname))
            encodings = fr.face_encodings(frame)

            while len(encodings) != 1:
                if lab == '0':
                    img_name = random.choice(
                        [x for x in glob.glob('train_data/0/*.jpg')])
                else:
                    img_name = random.choice(
                        [x for x in glob.glob('train_data/1/*.jpg')])

                frame = fr.load_image_file(img_name)
                encodings = fr.face_encodings(frame)

            data[idx, :] = encodings[0]
            label[idx] = int(lab)
            idx = idx + 1

        label = np.array(label, dtype=np.uint8)
        label = np.reshape(label, (len(label), 1))

        return data, label, lst
