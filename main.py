import os
import cv2
from pickled import *
from load_data import *
from argparse import ArgumentParser



parser = ArgumentParser()
parser.add_argument('--folder_path', default="train", help='choose a image folder')
parser.add_argument('--mode', default="train", help='train or test')
parser.add_argument('--read_image', default="ori_data", help='choose read_data or face_encoding')
args = parser.parse_args()




data_path = args.folder_path
file_list = 'image_list/image_{}_list.txt'.format(data_path)
save_path = './bin'
mode = args.mode


if __name__ == '__main__':
    if os.path.isfile(file_list):
        os.remove(file_list)
    imagelist(data_path, file_list)
    if args.read_image == 'ori_data':
        data, label, lst = read_data(file_list, data_path, shape=80)
    elif args.read_image == 'face_encoding':
        data, label, lst = face_encoding_read(file_list, data_path)
    pickled(save_path, data, label, lst, mode, args.read_image, data_path, bin_num=1)






