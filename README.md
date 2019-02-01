# face_classifier

## main.py
Writing train and test images in two txt files, such as image_train_list.txt and image_test_list.txt respectively. And access these two file in pickle when load data before traning. 
```
usage: main.py [--folder_path] [--mode] [--read_image]

optional arguments:
  --folder_path          choose a image folde
  --mode                 train or test
  --read_image           choose read_data or face_encoding
```
## pca_dim.py
Examing the result of PCA if it can be separable.
![image]https://github.com/Chung-Yi/face_classifier/blob/master/pca_data/save_image/output_layer_64.png
