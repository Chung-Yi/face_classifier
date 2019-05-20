import face_image
import numpy as np
import timeit
import os
import _pickle as cPickle
from keras.utils import np_utils
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from show_plot import *
from keras import regularizers
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras import backend as K
np.random.seed(10)

parser = ArgumentParser()
parser.add_argument('--output_layer', default=15, help='choose a image folder')
args = parser.parse_args()

output_layer = int(args.output_layer)


def extract_layer_output(model, train_image, output_layer):
    # extract specific layer output
    inp = model.input
    outputs = [layer.output for layer in model.layers]
    funct = K.function([inp, K.learning_phase()], outputs)
    # Testing
    train = train_image
    layer_outs = funct([train, 1.])
    layer_outs = layer_outs[output_layer]
    return layer_outs


def main():
    (x_train, y_train), (x_test, y_test) = face_image.load_data()
    y_labels = y_test

    # normalize
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    # one hot
    NUM = 2
    y_train = np_utils.to_categorical(y_train, NUM)
    y_test = np_utils.to_categorical(y_test, NUM)

    # model
    input_shape = x_train.shape[1:]

    model = Sequential()
    model.add(
        Conv2D(
            32,
            padding='same',
            kernel_size=(3, 3),
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01),
            input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(
        Conv2D(32, padding='same', kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(
        Conv2D(64, padding='same', kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(
        Conv2D(64, padding='same', kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(
        Conv2D(128, padding='same', kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(
        Conv2D(128, padding='same', kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    train_history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=200,
        verbose=2)

    model.save('model/face_classifier.h5')
    model.save_weights("model/face_classifier_weights.h5")

    idx = 0
    layer_output = extract_layer_output(model, x_train, output_layer)

    with open(
            os.path.join('pca_data', 'data_{}'.format(
                str(layer_output.shape[1]))), 'wb') as fi:
        cPickle.dump(layer_output, fi)

    show_train_history(train_history, 'acc', 'val_acc')
    show_train_history(train_history, 'loss', 'val_loss')

    loss, accuracy = model.evaluate(x_test, y_test)
    print('accuracy is %.4f' % (accuracy))

    start = timeit.default_timer()
    test_pred = model.predict_classes(x_test)
    end = timeit.default_timer()
    print('Time: ', end - start)

    predict_prop = model.predict(x_test)

    show_predict_prop(y_labels, test_pred, x_test, predict_prop, 13)
    plot_images_labels_prediction(x_test, y_labels, test_pred, 0, 10)

    # error image whose real label and predict label are different
    show_error_image(x_test, y_labels, test_pred)


if __name__ == '__main__':
    main()