import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import csv
import shutil
import numpy as np
import random
import math
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, set_start_method
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
import gc
from datetime import datetime
import platform

if platform.system() in 'Linux':
    from ffcv.loader import Loader, OrderOption
    from ffcv.fields.decoders import NDArrayDecoder
    from ffcv.transforms import ToTensor, ToDevice

def main():
    for i in range(NUMBER_OF_CHUNCKS)


if __name__ == '__main__':
    main()