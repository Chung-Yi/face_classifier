import tensorflow as tf
import timeit
import numpy as np
import cv2
from load_test_data import *
from keras.models import load_model
from show_plot import *

SHAPE = 80
CHANNEL_LEN = SHAPE * SHAPE
DATA_LEN = SHAPE * SHAPE * 3
c = CHANNEL_LEN

data = np.zeros((1, DATA_LEN), dtype=np.uint8)

# load model
model = load_model('model/face_classifier.h5')

face_img = cv2.imread('0_100810b3-b226-40ca-8d4b-83e0bc1ce9ec.jpg')
face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

face_img = cv2.resize(face_img, (80, 80))

face_img = np.reshape(face_img, (1, 80, 80, 3))
face_img = face_img.astype('float32') / 255

(x_other_test, y_other_test) = load_test_data()

(x_single_test, y_single_test) = load_single_test_data()

# predict multiple images
start = timeit.default_timer()
test_pred = model.predict_classes(x_other_test)
end = timeit.default_timer()
print('Time: ', end - start)

# predict single image
start = timeit.default_timer()
singel_test_pred = model.predict_classes(x_single_test)
end = timeit.default_timer()

predict_prop = model.predict(x_other_test)
single_predict_prop = model.predict(x_single_test)

show_predict_prop(y_other_test, test_pred, x_other_test, predict_prop, 13)
plot_images_labels_prediction(x_other_test, y_other_test, test_pred, 0, 10)

show_predict_prop(y_single_test, singel_test_pred, x_single_test,
                  single_predict_prop, 0)
plot_images_labels_prediction(x_single_test, y_single_test, singel_test_pred,
                              0, 1)

# error image whose real label and predict label are different
show_error_image(x_other_test, y_other_test, test_pred)
show_error_image(x_single_test, y_single_test, singel_test_pred)