import tensorflow as tf
from util import model_smaller_infer_first as model

import argparse

from model_classes_infer import standarUnetSquared_infer, aspectRatioVertical_infer, halfConvVertical_infer, biggerStridesVertical_infer
#######################################################################################
### $ python3 main_infer.py
#######################################################################################

def main(parser):

    graph = tf.Graph()
    with graph.as_default():

        model_unet = None
        # Create a model
        if parser.model_id is 1:
            model_unet = standarUnetSquared_infer.UNet(l2_reg=0.0001).model
        elif parser.model_id is 2:
            model_unet = aspectRatioVertical_infer.UNet(l2_reg=0.0001).model
        elif parser.model_id is 3:
            model_unet = halfConvVertical_infer.UNet(l2_reg=0.0001).model
        elif parser.model_id is 4:
            model_unet = biggerStridesVertical_infer.UNet(l2_reg=0.0001).model

        saver = tf.train.Saver(tf.global_variables())
        sess  = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print("in=", model_unet.inputs.name)
        print("on=", model_unet.outputs.name)

        saver.restore(sess, './models/deploy.ckpt')
        saver.save(sess, './models/deployfinal.ckpt')

        graphdef = graph.as_graph_def()
        tf.train.write_graph(graphdef, './models', 'semanticsegmentation_person.pbtxt', as_text=True)


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python main_infer.py',
        description='This module infer trained network',
        add_help=True
    )
    parser.add_argument('-m', '--model_id', type=int, default=1, help='Model ID')
    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    main(parser)

