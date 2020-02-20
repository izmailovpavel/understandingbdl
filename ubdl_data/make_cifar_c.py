# -*- coding: utf-8 -*-
# Copied from https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py

import os
import argparse
import warnings
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
import ctypes
from wand.api import library as wandlibrary
import numpy as np

from ubdl_data.corruptions import gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur
from ubdl_data.corruptions import zoom_blur, snow, frost, fog , brightness, contrast, elastic_transform, pixelate
from ubdl_data.corruptions import jpeg_compression, speckle_noise, gaussian_blur, spatter, saturate

warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--datapath', type=str, default=None, required=True, help='Path to uncorrupted data')
parser.add_argument('--savepath', type=str, default=None, required=True, help='Path for saving corrupted data')
args = parser.parse_args()



# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle

import collections

print('Using CIFAR-10 data')

d = collections.OrderedDict()
d['Gaussian Noise'] = gaussian_noise
d['Shot Noise'] = shot_noise
d['Impulse Noise'] = impulse_noise
d['Defocus Blur'] = defocus_blur
d['Glass Blur'] = glass_blur
d['Motion Blur'] = motion_blur
d['Zoom Blur'] = zoom_blur
d['Snow'] = snow
d['Frost'] = frost
d['Fog'] = fog
d['Brightness'] = brightness
d['Contrast'] = contrast
d['Elastic'] = elastic_transform
d['Pixelate'] = pixelate
d['JPEG'] = jpeg_compression

d['Speckle Noise'] = speckle_noise
d['Gaussian Blur'] = gaussian_blur
d['Spatter'] = spatter
d['Saturate'] = saturate


test_data = dset.CIFAR10(args.datapath, train=False)
convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])

os.makedirs(args.savepath, exist_ok=True)

for method_name in d.keys():
    print('Creating images for the corruption', method_name)
    cifar_c, labels = [], []

    for severity in range(1,6):
        corruption = lambda clean_img: d[method_name](clean_img, severity)

        for img, label in zip(test_data.data, test_data.targets):
            labels.append(label)
            cifar_c.append(np.uint8(corruption(convert_img(img))))

        np.savez(os.path.join(args.savepath, '{}_{}.npz'.format(d[method_name].__name__, severity)),
                data=np.array(cifar_c).astype(np.uint8),
                labels=np.array(labels).astype(np.uint8))

