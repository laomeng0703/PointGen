# PointGen
Work in progress.

The main idea comes from [this](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_PointAugment_An_Auto-Augmentation_Framework_for_Point_Cloud_Classification_CVPR_2020_paper.pdf)

Modified the model according to my personal thoughts. The feature extraction part uses dgcnn, and a new augmentation module is added to make the point cloud data more variable.
Please look forward to the completion of the paper.

### Dependencies

- Python 3.7
- CUDA 11
- PyTorch. At least 1.2.0
- (Optional) TensorboardX for visualization of the training process.

### Usage

Download the ModelNet40 dataset from [here.](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)

To train a model to classify point clouds sampled from 3D shapes:

    python train_PA.py --data_dir ModelNet40_Folder

Log files and network parameters will be saved to ```log``` folder in default.

Noted that the code may be not stable, if you can help please contact me.
