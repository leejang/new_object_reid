import h5py
import numpy as np
import scipy.misc

with h5py.File('mobilenet_v1_w_sa_two_self_sn_n_cos_loss_visualize_attention_maps_features.h5', 'r') as f:
  dataset = f['att_map_2']
  # img0
  #data = dataset[0, 14*22, :]
  # img1
  #data = dataset[1, 11*1, :]
  # img2
  data = dataset[2, 18*12, :]
  print (data.shape)

  arr_to_save = np.array(data)
  arr_to_save = np.reshape(arr_to_save, (28, 28))
  #print (arr_to_save.shape)

  #scipy.misc.imsave('img0_14_22_att_map.jpg', arr_to_save)
  #scipy.misc.imsave('img1_11_1_att_map.jpg', arr_to_save)
  scipy.misc.imsave('img2_18_12_att_map.jpg', arr_to_save)
