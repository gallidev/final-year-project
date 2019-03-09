import sys, time
import numpy as np
import tensorflow as tf
from PIL import Image
from util import loader as ld
import math


def load_dataset(train_rate):
    #loader = ld.Loader(dir_original="data_set/VOCdevkit/person/JPEGImages",
    #                   dir_segmented="data_set/VOCdevkit/person/SegmentationClass")
    loader = ld.Loader(dir_original="data_set/portraits/images_jpg",
                        dir_segmented="data_set/portraits/masks_png")
    #loader = ld.Loader(dir_original="data_set/portraits/test_image",
    #                   dir_segmented="data_set/portraits/test_mask")
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

if __name__ == '__main__':

    # Load train and test datas
    train, test = load_dataset(train_rate=0.85)

    np.set_printoptions(threshold=sys.maxsize)

    palette = [0, 0, 0, 255, 255, 255]

    prepimg = test.images_original[0][np.newaxis, :, :, :]

    totalAccuracy = 0.0

    # Segmentation prepare tflite interpreter
    interpreter = tf.contrib.lite.Interpreter(model_path="model/128-MyModel_26Epochs_test/128_portraits_26ep_32ba_quantized_32.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for idx, image in enumerate(test.images_original):
        prepimg = image[np.newaxis, :, :, :]
        interpreter.reset_all_variables()
        interpreter.set_tensor(input_details[0]['index'], np.array(prepimg, dtype=np.float32))
        interpreter.invoke()
        outputs = interpreter.get_tensor(output_details[0]['index'])
        # View
        output = outputs[0]

        seg_image = create_png_from_array(test.images_segmented[idx], palette)
        # seg_image.show();

        result = create_png_from_array(output, palette)
        # result.show();

        iou_score = calculate_accuracy(result, seg_image)
        if(not(math.isnan(iou_score))):
            print("iou_Score:" + str(iou_score))
            totalAccuracy += iou_score
        if( idx % 100 == 0):
            print("done: " + str(idx+1))

        print(totalAccuracy)

    print(totalAccuracy / len(test.images_original))
    print("The accuracy overall is: " + str(totalAccuracy / len(test.images_original)))



