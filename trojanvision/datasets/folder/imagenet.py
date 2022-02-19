#!/usr/bin/env python3

from trojanvision.datasets.imagefolder import ImageFolder
from trojanzoo.utils.module import Module

from torchvision import datasets
import os
import json

from trojanvision import __file__ as root_file
root_dir = os.path.dirname(root_file)


class ImageNet(ImageFolder):
    r"""ImageNet dataset. It inherits :class:`trojanvision.datasets.ImageFolder`.

    Note:
        According to https://github.com/pytorch/vision/issues/1563,
        You need to personally visit https://image-net.org/download-images.php
        to download the dataset.

        Expected files:

            * ``'{self.folder_path}/ILSVRC2012_devkit_t12.tar.gz'``
            * ``'{self.folder_path}/ILSVRC2012_img_train.tar'``
            * ``'{self.folder_path}/ILSVRC2012_img_val.tar'``
            * ``'{self.folder_path}/meta.bin'``

    See Also:
        :any:`torchvision.datasets.ImageNet`

    Attributes:
        name (str): ``'imagenet'``
        num_classes (int): ``1000``
        data_shape (list[int]): ``[3, 224, 224]``
        norm_par (dict[str, list[float]]):
            | ``{'mean': [0.485, 0.456, 0.406],``
            | ``'std'  : [0.229, 0.224, 0.225]}``
    """

    name = 'imagenet'
    url = {
        'train': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar',
        'valid': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar',
        'test': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar',
    }
    md5 = {
        'train': '1d675b47d978889d74fa0da5fadfb00e',
        'valid': '29b22e2961454d5413ddabcf34fc5622',
        'devkit': 'fa75699e90414af021442c21a62c3abf',
    }

    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [0.485, 0.456, 0.406],
                                                           'std': [0.229, 0.224, 0.225], },
                 **kwargs):
        super().__init__(norm_par=norm_par, **kwargs)

    def initialize_folder(self):
        try:
            datasets.ImageNet(root=self.folder_path,
                              split='train', download=True)
            datasets.ImageNet(root=self.folder_path, split='val', download=True)
        except RuntimeError:
            raise RuntimeError('\n\n'
                               'You need to visit \'https://image-net.org/download-images.php\' '
                               'to download ImageNet.\n'
                               'There are direct links to files, but not legal to distribute. '
                               'Please apply for access permission and find links yourself.\n\n'
                               f'folder_path: {self.folder_path}\n'
                               'expected files:\n'
                               '{folder_path}/ILSVRC2012_devkit_t12.tar.gz\n'
                               '{folder_path}/ILSVRC2012_img_train.tar\n'
                               '{folder_path}/ILSVRC2012_img_val.tar\n'
                               '{folder_path}/meta.bin')
        os.symlink(os.path.join(self.folder_path, 'imagenet', 'val'),
                   os.path.join(self.folder_path, 'imagenet', 'valid'))


class Sample_ImageNet(ImageNet):

    name: str = 'sample_imagenet'
    num_classes = 10
    url = {}
    org_folder_name = {}

    def initialize_folder(self):
        _dict = Module(self.__dict__)
        _dict.__delattr__('folder_path')
        imagenet = ImageNet(**_dict)
        class_dict: dict = {}
        json_path = os.path.normpath(os.path.join(
            root_dir, 'data', 'sample_imagenet', 'class_dict.json'))
        with open(json_path, 'r', encoding='utf-8') as f:
            class_dict: dict = json.load(f)
        imagenet.sample(child_name=self.name, class_dict=class_dict)
