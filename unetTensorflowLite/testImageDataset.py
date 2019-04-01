
import argparse
import random
import tensorflow as tf
import sys


from util import loader as ld
from util import model as model
from util import repoter as rp



loader = ld.Loader(dir_original="data_set/portraits/test_image",
                       dir_segmented="data_set/portraits/test_mask")

#ld.image_generator(paths_segmented, init_size, normalization=False)

