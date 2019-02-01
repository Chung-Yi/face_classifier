import os
import pickle
import _pickle as cPickle





def pickled(savepath, data, label, fnames, mode, read_image, data_path, bin_num=None):
  assert os.path.isdir(savepath)
  total_num = len(fnames) 
  samples_per_bin = total_num / bin_num 
  assert samples_per_bin > 0
  idx = 0
  for i in range(bin_num): 
    start = int(i*samples_per_bin) 
    end = int((i+1)*samples_per_bin) 
    
    if end <= total_num:
      dic = {'data': data[start:end, :],
              'labels': label[start:end],
              'filenames': fnames[start:end]}
    else:
      dic = {'data': data[start:, :],
              'labels': label[start:],
              'filenames': fnames[start:]}
    if mode == "train":
      dic['batch_label'] = "training batch {} of {}".format(idx, bin_num)
    else:
      dic['batch_label'] = "testing batch {} of {}".format(idx, bin_num)
      
    with open(os.path.join(savepath, 'data_{}_{}_{}_'.format(data_path, read_image, mode)+str(idx)), 'wb') as fi:
      cPickle.dump(dic, fi)
    idx = idx + 1

