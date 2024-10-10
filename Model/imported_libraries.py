import numpy as np

from tqdm import tqdm

from __future__ import print_function

import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import uniform_filter1d

from six.moves import xrange

import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from torchvision.utils import save_image, make_grid 
from torchvision.datasets import MNIST
from torch.optim import Adam


dataset_path = '~/datasets'

cuda = True 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
print(device)
