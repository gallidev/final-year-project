import sys, time
import numpy as np
import tensorflow as tf
from PIL import Image
from util import loader as ld
import math
import argparse


def load_dataset(train_rate, init_size, squared, test=False):

    loader = None

    if test:
        loader = ld.Loader(dir_original="data_set/portraits/test_image",
                        dir_segmented="data_set/portraits/test_mask",
                        init_size=init_size,
                        squared=squared)
    else:
        loader = ld.Loader(dir_original="data_set/portraits/images_jpg",
                        dir_segmented="data_set/portraits/masks_png",
                        init_size=init_size,
                        squared=squared)

    return loader.load_train_test(train_rate=train_rate, shuffle=False)


def calculate_accuracy(output, ground_truth):
    # Check accuracy
    intersection = np.logical_and(output, ground_truth)
    union = np.logical_or(output, ground_truth)
    return np.sum(intersection) / np.sum(union)


# Create Pil image from the output as ndarray in float of 128x128x2
def create_png_from_array(array, palette):
    array = np.argmax(array, axis=2)
    image = Image.fromarray(np.uint8(array), mode="P")
    image.putpalette(palette)
    return image

def evaluate(args):
    # Load train and test datas
    init_size = tuple(args.init_size)
    print(init_size)
    print("Squared: " + str(args.squared) )
    print("tflite: " + str(args.tflite) )
    print("test: " + str(args.test) )
    train, test = load_dataset(train_rate=args.train_rate, init_size=init_size, squared=args.squared, test=args.test)

    np.set_printoptions(threshold=sys.maxsize)

    palette = [0, 0, 0, 255, 255, 255]

    totalAccuracy = 0.0
    totalNan = 0

    # Segmentation prepare tflite interpreter
    interpreter = None
    input_details = None
    output_details = None
    sess = None

    if(args.tflite):   
        interpreter = tf.contrib.lite.Interpreter(model_path=args.model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    else:
        with tf.gfile.FastGFile(args.model_path, "rb") as f:
            graphdef = tf.GraphDef()
            graphdef.ParseFromString(f.read())
            _ = tf.import_graph_def(graphdef, name="")
        sess = tf.Session()

    for idx, image in enumerate(test.images_original):
        prepimg = image[np.newaxis, :, :, :]
        outputs = []

        if(args.tflite): 
            interpreter.reset_all_variables()
            interpreter.set_tensor(input_details[0]['index'], np.array(prepimg, dtype=np.float32))
            interpreter.invoke()
            outputs = interpreter.get_tensor(output_details[0]['index'])
        else:
            outputs = sess.run("output/BiasAdd:0", {"input:0": prepimg})

        # View
        output = outputs[0]

        seg_image = create_png_from_array(test.images_segmented[idx], palette)
        seg_image.show();

        result = create_png_from_array(output, palette)
        result.show();

        iou_score = calculate_accuracy(result, seg_image)
        if(not(math.isnan(iou_score))):
            #print("iou_Score:" + str(iou_score))
            totalAccuracy += iou_score
        else:
            totalNan += 1
        if( idx % 100 == 0):
            print("done: " + str(idx+1))

        #print(totalAccuracy)

    print("The model " + args.model_path + " accuracy overall is: " +
        str(totalAccuracy / (len(test.images_original) - totalNan)))


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Evaluation of tflite models',
        usage='python3 tflite_test_accuracy.py ',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-m', '--model_path', help='Model path')
    parser.add_argument('-t', '--train_rate', type=float, default=0.85, help='Training rate used')
    parser.add_argument('-i', '--init_size', nargs='+', type=int,  default=[128, 128], help='Size of the model')
    parser.add_argument('-s', '--squared',  help='Square the input image', dest='squared', action='store_true')
    parser.add_argument('-ns', '--no_squared', help='Do not square the input image', dest='squared', action='store_false')
    parser.set_defaults(squared=True)
    parser.add_argument('-tf', '--tflite',  help='tflite model', dest='tflite', action='store_true')
    parser.add_argument('-notf', '--no_tflite', help='pb model', dest='tflite', action='store_false')
    parser.set_defaults(tflite=True)
    parser.add_argument('-te', '--test',  help='test data', dest='test', action='store_true')
    parser.set_defaults(test=False)
    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    evaluate(parser)


