###################################################################
# $ python3 main.py --gpu --augmentation --batchsize 32 --epoch 50
###################################################################

import argparse
import random
import tensorflow as tf
import sys

from util import loader as ld
from util import model_smaller_first as model
from util import repoter as rp


# Create a model
model_unet = model.UNet(l2_reg=0.0001).model

correct_prediction = tf.equal(tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3))

