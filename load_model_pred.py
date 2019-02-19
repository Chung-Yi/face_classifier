import tensorflow as tf
import timeit
from load_test_data import *
from keras.models import load_model
from show_plot import *

# load model
model = load_model('model/face_classifier.h5')

(x_other_test, y_other_test) = load_test_data()

(x_single_test, y_single_test) = load_single_test_data()
print(x_single_test.shape, y_single_test.shape)


# predict multiple images
start = timeit.default_timer()
test_pred = model.predict_classes(x_other_test)
end = timeit.default_timer()
print('Time: ', end - start)

# predict single image
start = timeit.default_timer()
singel_test_pred = model.predict_classes(x_single_test)
end = timeit.default_timer()
print('Time: ', end - start)


predict_prop = model.predict(x_other_test)
single_predict_prop = model.predict(x_single_test)
    
show_predict_prop(y_other_test, test_pred, x_other_test, predict_prop, 13)
plot_images_labels_prediction(x_other_test, y_other_test, test_pred, 0, 10)

show_predict_prop(y_single_test, singel_test_pred, x_single_test, single_predict_prop, 0)
plot_images_labels_prediction(x_single_test, y_single_test, singel_test_pred, 0, 1)


# error image whose real label and predict label are different
show_error_image(x_other_test, y_other_test, test_pred)
show_error_image(x_single_test, y_single_test, singel_test_pred)