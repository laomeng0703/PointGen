import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Augmentor import Augmentor
from utils import point_operation, data_utils as d_utils
from main_parameter import args

from torchvision import transforms

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

point_transform = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRotate(),
        d_utils.PointcloudRotatePerturbation(),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter(),
    ]
)

# def translate_pointcloud(pointcloud):
#     xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
#     xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
#
#     translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
#     return translated_pointcloud

class Modelnetload(Dataset):
    def __init__(self, args, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = args.num_points
        self.partition = partition
        self.Augmentor = Augmentor(args) if args.Augmentor else None

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            np.random.shuffle(pointcloud)

            if self.Augmentor is not None:
                origin, pointcloud = self.Augmentor(pointcloud)
            pointcloud = point_operation.rotate_point_cloud_and_gt(pointcloud)
            pointcloud = point_operation.jitter_perturbation_point_cloud(pointcloud)

        return pointcloud.astype(np.float32), label.astype(np.int32)

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_TEST_DIR = os.path.join(BASE_DIR, 'data_test')
    train = Modelnetload(args)

    #test = Modelnetload(1024)
    for data, label in train:
        for i in range(10):
            fb = np.savetxt(DATA_TEST_DIR + '%06d.txt'%(i), data)




