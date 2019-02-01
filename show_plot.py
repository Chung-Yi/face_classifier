import matplotlib.pyplot as plt
import numpy as np

label_dict = {0:'bad', 1:'good'}

def show_train_history(train_history, train, validation):
  plt.plot(train_history.history[train])
  plt.plot(train_history.history[validation])
  plt.title('Train history')
  plt.ylabel(train)
  plt.xlabel('Epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()

def show_predict_prop(y, prediction, x_img_test, predict_prop, i):
  print('label:', label_dict[y[i][0]], 'predict', label_dict[np.argmax(predict_prop[i])])
  plt.figure(figsize=(2,2))
  plt.imshow(np.reshape(x_img_test[i], (80,80,3)))
  plt.show()
  for j in range(2):
    print(label_dict[j]+'Probability:%1.9f'%(predict_prop[i][j]))


def show_error_image(x_test, y_labels, test_pred):
    err_indexes = []
    for i, image in enumerate(x_test):
        if test_pred[i] != y_labels[i][0]:
            err_indexes.append(i)
    
    for i in range(len(err_indexes)):
        idx = err_indexes[i]
        print(idx)
        plt.subplot(3, 3, i+1)
        plt.imshow(x_test[idx], cmap='binary')
        plt.title('p:{}|{}'.format(label_dict[test_pred[idx]], label_dict[y_labels[idx][0]]))
    plt.show()

def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
  fig = plt.gcf()
  fig.set_size_inches(12, 14)
  if num > 25:
    num = 25
  for i in range(num):
    ax = plt.subplot(5, 5, i+1)
    ax.imshow(images[idx], cmap='binary')

    title = str(i)+','+label_dict[labels[idx][0]]
    if len(prediction) > 0:
      title+='=>'+label_dict[prediction[idx]]
    
    
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    idx+=1
  plt.show()