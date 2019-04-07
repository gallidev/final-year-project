###################################################################
# $ python3 main.py --gpu --batchsize 32 --epoch 20 --init_size 128 128 --squared
###################################################################

import argparse
import random
import tensorflow as tf
import sys

from util import loader as ld
from model_classes import standardUnetSquared, aspectRatioVertical, halfConvVertical, biggerStridesVertical
from util import repoter as rp


def load_dataset(train_rate, init_size, squared, test=False, augmentation=False):

    loader = None

    folderJpegs = "images_jpg"
    folderMasks = "masks_png"

    if augmentation:
        folderJpegs = "JPEGImagesOUT"
        folderMasks = "SegmentationClassOUT"
    if test:
        folderJpegs = "test_image"
        folderMasks = "test_mask"

    loader = ld.Loader(dir_original="data_set/portraits/" + folderJpegs,
                        dir_segmented="data_set/portraits/" + folderMasks,
                        init_size=init_size,
                        squared=squared)

    return loader.load_train_test(train_rate=train_rate, shuffle=False)

def train(parser):

    init_size = tuple(parser.init_size)
    if parser.model_id is not 1:
        init_size = [96, 128]
        parser.squared = False

    print(init_size)
    print("Model ID: " + str(parser.model_id))
    print("Squared: " + str(parser.squared))
    print("test: " + str(parser.test))
    train, test = load_dataset(train_rate=parser.trainrate, init_size=init_size,
                               squared=parser.squared, test=parser.test, augmentation=parser.augmentation)

    valid = train.perm(0, 30)
    test = test.perm(0, 150)

    # Create Reporter Object
    reporter = rp.Reporter(parser=parser)
    accuracy_fig = reporter.create_figure("Accuracy", ("epoch", "accuracy"), ["train", "test"])
    loss_fig = reporter.create_figure("Loss", ("epoch", "loss"), ["train", "test"])

    # Whether or not using a GPU
    gpu = parser.gpu

    model_unet = None
    # Create a model
    if parser.model_id is 1:
        model_unet = standardUnetSquared.UNet(l2_reg=parser.l2reg).model
    elif parser.model_id is 2:
        model_unet = aspectRatioVertical.UNet(l2_reg=parser.l2reg).model
    elif parser.model_id is 3:
        model_unet = halfConvVertical.UNet(l2_reg=parser.l2reg).model
    elif parser.model_id is 4:
        model_unet = biggerStridesVertical.UNet(l2_reg=parser.l2reg).model

    # Set a loss function and an optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_unet.teacher,
                                                                           logits=model_unet.outputs))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize session
    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7), device_count={'GPU': 1},
                                log_device_placement=False, allow_soft_placement=True)
    saver = tf.train.Saver()
    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train the model
    epochs = parser.epoch
    batch_size = parser.batchsize
    #is_augment = parser.augmentation
    is_augment = False
    train_dict = {model_unet.inputs: valid.images_original, model_unet.teacher: valid.images_segmented,
                  model_unet.is_training: False}
    test_dict = {model_unet.inputs: test.images_original, model_unet.teacher: test.images_segmented,
                 model_unet.is_training: False}

    for epoch in range(epochs):
        for batch in train(batch_size=batch_size, augment=is_augment):
            # Expansion of batch data
            inputs = batch.images_original
            teacher = batch.images_segmented
            # Training
            sess.run(train_step, feed_dict={model_unet.inputs: inputs, model_unet.teacher: teacher,
                                            model_unet.is_training: True})

        # Evaluation
        if epoch % 1 == 0:
            loss_train = sess.run(cross_entropy, feed_dict=train_dict)
            loss_test = sess.run(cross_entropy, feed_dict=test_dict)
            accuracy_train = sess.run(accuracy, feed_dict=train_dict)
            accuracy_test = sess.run(accuracy, feed_dict=test_dict)
            print("Epoch:", epoch)
            print("[Train] Loss:", loss_train, " Accuracy:", accuracy_train)
            print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)
            accuracy_fig.add([accuracy_train, accuracy_test], is_update=True)
            loss_fig.add([loss_train, loss_test], is_update=True)
            if epoch % 3 == 0:
                idx_train = random.randrange(10)
                idx_test = random.randrange(100)
                outputs_train = sess.run(model_unet.outputs,
                                         feed_dict={model_unet.inputs: [train.images_original[idx_train]],
                                                    model_unet.is_training: False})
                outputs_test = sess.run(model_unet.outputs,
                                        feed_dict={model_unet.inputs: [test.images_original[idx_test]],
                                                   model_unet.is_training: False})
                train_set = [train.images_original[idx_train], outputs_train[0], train.images_segmented[idx_train]]
                test_set = [test.images_original[idx_test], outputs_test[0], test.images_segmented[idx_test]]
                reporter.save_image_from_ndarray(train_set, test_set, train.palette, epoch,
                                                 index_void=len(ld.DataSet.CATEGORY)-1)
        saver.save(sess, './models/deploy.ckpt')
        print("in=", model_unet.inputs.name)
        print("on=", model_unet.outputs.name)

    # Test the trained model
    loss_test = sess.run(cross_entropy, feed_dict=test_dict)
    accuracy_test = sess.run(accuracy, feed_dict=test_dict)
    print("Result")
    print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python main.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPUs')
    parser.add_argument('-m', '--model_id', type=int, default=1, help='Model ID')
    parser.add_argument('-e', '--epoch', type=int, default=250, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=32, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float, default=0.85, help='Training rate')
    parser.add_argument('-a', '--augmentation', action='store_true', help='Number of epochs')
    parser.add_argument('-r', '--l2reg', type=float, default=0.0001, help='L2 regularization')
    parser.add_argument('-i', '--init_size', nargs='+', type=int,  default=[128, 128], help='Size of the model')
    parser.add_argument('-s', '--squared',  help='Square the input image', dest='squared', action='store_true')
    parser.add_argument('-ns', '--no_squared', help='Do not square the input image', dest='squared', action='store_false')
    parser.set_defaults(squared=True)
    parser.add_argument('-te', '--test',  help='test data', dest='test', action='store_true')
    parser.set_defaults(test=False)

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
