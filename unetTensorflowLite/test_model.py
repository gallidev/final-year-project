###################################################################
# $ python3 main.py --gpu --augmentation --batchsize 32 --epoch 50
###################################################################

import argparse
import random
import tensorflow as tf
import sys

from model_classes import standardUnetSquared, aspectRatioVertical, standardUnetSquared


# Create a model
model_unet = standardUnetSquared.UNet(l2_reg=0.0001).model

correct_prediction = tf.equal(tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3))

