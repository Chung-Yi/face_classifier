import matplotlib.pyplot as plt
import _pickle as cPickle
import numpy as np
import utils
from sklearn.decomposition import PCA



f = open('image_train_list.txt')
lines = f.read().splitlines()




def save_chart(features, labels, title='', txt=''):
    fig = plt.figure(figsize=(12, 6))
    utils.draw_chart(features, labels, title, txt)
    fig.savefig('pca_data/save_image/'+title, bbox_inches='tight')


def main():
    idx = 0
    label = [int(ln.split(' ')[1]) for ln in lines]
    
    for ln in lines:
        lab = ln.split(' ')[1]
        label[idx] = int(lab)
        idx += 1
    labels = np.array(label, dtype=np.uint8)
    labels = np.reshape(label, (len(label), 1))

    with open('pca_data/data_256', 'rb') as f:
        features = cPickle.load(f)

    pca = PCA(n_components=2)
    features = pca.fit_transform(features)
    utils.draw_chart(features, labels, title='PCA')
    save_chart(features, labels, title='output_layer_256')



    
    


    





if __name__ == '__main__':
    main()