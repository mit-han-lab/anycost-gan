from torchvision import datasets, transforms
import torchvision.transforms.functional as F
from PIL import Image
import random


class NativeDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(NativeDataset, self).__getitem__(index)[0]  # only return the image


class MultiResize(object):
    def __init__(self, highest_res, n_res=4, interpolation=Image.BILINEAR):
        all_res = []
        for _ in range(n_res):
            all_res.append(highest_res)
            highest_res = highest_res // 2
        all_res = sorted(all_res)  # always low to high
        self.transforms = [transforms.Resize(r, interpolation) for r in all_res]

    def __call__(self, img):
        return [t(img) for t in self.transforms]


class GroupRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return [F.hflip(i) for i in img]
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class GroupTransformWrapper(object):
    # applying the same transform (no randomness) to each of the images in a list
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        return [self.transform(i) for i in img]

    def __repr__(self):
        return self.__class__.__name__
