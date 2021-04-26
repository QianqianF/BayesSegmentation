import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
from pathlib import Path

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]
# The following two functions are copied from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


classes = [
    "Nuclei",
    "Void",
]

# can't verify below?
# https://github.com/yandex/segnet-torch/blob/master/datasets/camvid-gen.lua
""" class_weight = torch.FloatTensor([
    0.58872014284134, 0.51052379608154, 2.6966278553009,
    0.45021694898605, 1.1785038709641, 0.77028578519821, 2.4782588481903,
    2.5273461341858, 1.0122526884079, 3.2375309467316, 4.1312313079834, 0]) """
class_weight = torch.FloatTensor(
    [1.0, 1.0]
)

# mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
# std = [0.27413549931506, 0.28506257482912, 0.28284674400252]

class_color = [
    (255, 255, 255),
    (0, 0, 0),
]


def _make_dataset(dir):
    images = []
    labels = []
    for f in os.scandir(dir): 
        if os.path.isdir(f.path):
            image = os.path.join(f.path, 'images')
            item = next(os.scandir(image)).path
            images.append(item)

            # combine individual masks into one mask if it's not in the dataset
            if not os.path.exists(os.path.join(f.path, 'mask')):
                os.makedirs(os.path.join(f.path, 'mask'))

                im = Image.open(item)
                mask = np.zeros((im.size[1], im.size[0]))

                masks = os.path.join(f.path, 'masks')
                for m in os.scandir(masks):
                    if is_image_file(m.path):
                        one_cell = np.array(Image.open(m.path))
                        if mask.shape != one_cell.shape:
                            print(m.path)
                        mask += one_cell

                mask = Image.fromarray(mask).convert('L')
                mask.save(os.path.join(f.path, 'mask', 'mask.png'))
            
            labels.append(next(os.scandir(os.path.join(f.path, 'mask'))).path)
        
        
    return images, labels


# class LabelTensorToPILImage(object):
#     def __call__(self, label):
#         label = label.unsqueeze(0)
#         colored_label = torch.zeros(3, label.size(1), label.size(2)).byte()
#         for i, color in enumerate(class_color):
#             mask = label.eq(i)
#             for j in range(3):
#                 colored_label[j].masked_fill_(mask, color[j])
#         npimg = colored_label.numpy()
#         npimg = np.transpose(npimg, (1, 2, 0))
#         mode = None
#         if npimg.shape[2] == 1:
#             npimg = npimg[:, :, 0]
#             mode = "L"

#         return Image.fromarray(npimg, mode=mode)


class Nuclei(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        joint_transform=None,
        transform=None,
        target_transform=None,
        download=False,
        loader=default_loader,
    ):
        self.root = root
        # self.root = Path(root)

        assert split in ("train", "val", "test")
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        self.class_weight = class_weight
        self.classes = classes
        # self.mean = mean
        # self.std = std

        if download:
            self.download()

        # print(type(self.root))
        self.imgs, self.labels = _make_dataset(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        img_path, target_path = self.imgs[index], self.labels[index]
        img = self.loader(img_path)
        target = self.loader(target_path)

        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        print(img.size, target.size)
        return img, target

    def __len__(self):
        return len(self.imgs)

    def download(self):
        # TODO: please download the dataset from
        # https://www.kaggle.com/c/data-science-bowl-2018/overview
        raise NotImplementedError
