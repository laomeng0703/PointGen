import h5py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048')

f = h5py.File(DATA_DIR + '/ply_data_train4.h5', 'r')
f.keys()
print(f.keys())

for key in f.keys():
    print(key)
    print(f[key].name)
    print(f[key].shape)
