import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import random
import sys, gc, os, logging
from tqdm import tqdm
import pickle
import wandb
import datetime

import cv2, matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import albumentations
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as vision_models
import torch.optim as optim
from scipy.spatial.distance import cdist
import cupy as cp

from efficientnet_pytorch import EfficientNet

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
IMAGE_SIZE = (384, 384)
TEXT_VEC_SIZE = 15000
data_folder = "data"