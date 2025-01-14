import os

import numpy as np
import torch.utils.data as data
from PIL import Image

"""# Load Dataset"""


class Dataset(data.Dataset):

    def __init__(self, root='./', load_set='train', transform=None):
        self.root = root
        self.transform = transform
        self.load_set = load_set  # 'train','val','test'

        self.images = np.load(os.path.join(root, f'images-{self.load_set}.npy'))
        self.points2d = np.load(os.path.join(root, f'points2d-{self.load_set}.npy'))
        self.points3d = np.load(os.path.join(root, f'points3d-{self.load_set}.npy'))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """

        image = Image.open(self.images[index])
        point2d = self.points2d[index]
        point3d = self.points3d[index]

        if self.transform is not None:
            image = self.transform(image)

        return image[:3], point2d, point3d

    def __len__(self):
        return len(self.images)
