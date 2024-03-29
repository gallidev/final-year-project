import tensorflow as tf
from util import loader as ld


class UNet:
    def __init__(self, 
            #size=(256, 256), 
            size=(128, 96),
            l2_reg=None):
        self.model = self.create_model(size, l2_reg)

    @staticmethod
    def create_model(size, l2_reg):
        inputs = tf.placeholder(tf.float32, [None, size[0], size[1], 3], name="input")

        # 128x128x3
        conv1_1 = UNet.conv(inputs, filters=16, l2_reg_scale=l2_reg, batchnorm_istraining=False)
        # 64x64x16
        pool1 = UNet.pool(conv1_1)
        # 32x32x16

        conv2_1 = UNet.conv(pool1, filters=32, strides=2, l2_reg_scale=l2_reg, batchnorm_istraining=False)
        # 32x32x32
        pool2 = UNet.pool(conv2_1)
        # 16x16x32

        conv3_1 = UNet.conv(pool2, filters=64, strides=2, l2_reg_scale=l2_reg, batchnorm_istraining=False)
        # 8x8x64
        pool3 = UNet.pool(conv3_1)
        # 4x4x64

        conv4 = UNet.conv(pool3, filters=128, strides=2, l2_reg_scale=l2_reg)
        # 2x2x128
        concated1 = tf.concat([UNet.conv_transpose(conv4, filters=64, strides=[4, 3], l2_reg_scale=l2_reg), conv3_1],
                              axis=3)
        # 8x8x64

        conv_up1_1 = UNet.conv(concated1, filters=64, l2_reg_scale=l2_reg)
        # 8x8x64
        concated2 = tf.concat(
            [UNet.conv_transpose(conv_up1_1, filters=32, strides=[4, 4], l2_reg_scale=l2_reg), conv2_1], axis=3)
        # 32x32x32

        conv_up2_1 = UNet.conv(concated2, filters=32, l2_reg_scale=l2_reg)
        # 32x32x32
        concated3 = tf.concat(
            [UNet.conv_transpose(conv_up2_1, filters=16, strides=[4, 4], l2_reg_scale=l2_reg), conv1_1], axis=3)
        # 128x128x16

        conv_up3_1 = UNet.conv(concated3, filters=16, l2_reg_scale=l2_reg)
        # 128x128x16

        outputs = UNet.conv(conv_up3_1, filters=ld.DataSet.length_category(), kernel_size=[1, 1], activation=None,
                            name="output")
        # 128x128x2

        return Model(inputs, outputs, None, False)

    @staticmethod
    def conv(inputs, filters, kernel_size=[3, 3], strides=1, activation=tf.nn.relu, l2_reg_scale=None,
             batchnorm_istraining=None, name=None):
        if l2_reg_scale is None:
            regularizer = None
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
        conved = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer,
            name=name
        )
        if batchnorm_istraining is not None:
            conved = UNet.bn(conved, batchnorm_istraining)

        return conved

    @staticmethod
    def bn(inputs, is_training):
        normalized = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            momentum=0.9,
            epsilon=0.001,
            center=True,
            scale=True,
            training=is_training,
        )
        return normalized

    @staticmethod
    def pool(inputs):
        pooled = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)
        return pooled

    @staticmethod
    def conv_transpose(inputs, filters, strides=[2, 2], l2_reg_scale=None):
        if l2_reg_scale is None:
            regularizer = None
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
        conved = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=filters,
            strides=strides,
            kernel_size=[2, 2],
            padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer
        )
        return conved


class Model:
    def __init__(self, inputs, outputs, teacher, is_training):
        self.inputs = inputs
        self.outputs = outputs
        self.teacher = teacher
        self.is_training = is_training
