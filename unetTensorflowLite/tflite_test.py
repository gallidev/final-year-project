import sys, time
import numpy as np
import tensorflow as tf
from PIL import Image


def make_square(im, imageMode, init_size=(128,128), fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(init_size[0], x, y)
    #check if the image is a png it won't accept the color black in 4 values
    if(imageMode == 'L' or imageMode == 'P'):
        fill_color = 0
    new_im = Image.new(imageMode, (size, size), fill_color)
    new_im.paste(im, ((size - x) // 2, (size - y) // 2))
    return new_im

# Crop a bit to avoid the rotation black part
def crop_center(img, cropx, cropy):

    centerx = (img.width - cropx) / 2
    centery = (img.width - cropy) / 2

    left, upper = centerx, centery

    right, bottom = img.width - centerx, img.height - centery

    return img.crop((left, upper, right, bottom))


def calculate_accuracy(output, ground_truth):
    # Check accuracy
    intersection = np.logical_and(output, ground_truth)
    union = np.logical_or(output, ground_truth)
    return np.sum(intersection) / np.sum(union)


def get_only_human(img, mask, background):
    img = img.convert("RGBA")
    imgDatas = img.getdata()
    maskDatas = mask.getdata()
    backgroundData = background.getdata()

    newData = []
    for idx, item in enumerate(maskDatas):
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append(imgDatas[idx])
        else:
            newData.append(backgroundData[idx])

    img.putdata(newData)
    return img


models = [
        ["6_model", "/6_model_32e_96_128_aug_quantized.tflite", "", False],
        ["5_model", "/5_model_32e_96_128_aug_quantized.tflite", "", False],
        ["4_model", "/4_model_12e_96_128_aug_quantized.tflite", "", False],
        ["premade", "/semanticsegmentation_frozen_person_quantized_32.tflite", "", True],
        ["1_model", "/1_model_20e_128_aug_quantized.tflite", "aug_", True],
        ["1_model", "/1_model_20e_128_quantized.tflite", "", True],
        ]

if __name__ == '__main__':

    # read images
    images = [Image.open("images/test1.jpg"), Image.open("images/test2.jpg"), Image.open("images/test3.jpg"),
              Image.open("images/test4.jpg"), Image.open("images/test5.jpg")]
    imagesMasks = [Image.open("images/mask1.png"), Image.open("images/mask2.png"), Image.open("images/mask3.png"),
                   Image.open("images/mask4.png"), Image.open("images/mask5.png")]

    backgroundImage = Image.open("images/testBackgroundDarker.png")

    #save input images only human with ground truth
    for idx, inputImage in enumerate(images):
        maskRGB = imagesMasks[idx].convert("RGB")
        #maskRGB = maskRGB.resize((48, 64))
        maskRGB = maskRGB.resize((600, 800))
        #maskRGB.show()
        groundTruth = get_only_human(inputImage, maskRGB, backgroundImage)
        groundTruth.save("images/groundtruth" + str(idx+1) + ".png")


    imageMask = Image.open("1.jpg")
    imageMaskSquared = make_square(imageMask, imageMask.mode)
    imageMaskSquared.save("Squared1.jpg")


    for model in models:

        model_folder = model[0]

        # model path
        model_path = "models/" + model_folder + model[1]

        square = model[3]

        relativePathResults = "models/" + model_folder + "/results/" + model_folder + "_" + model[2]

        index = 1
        for idx, inputImage in enumerate(images):
            print(str(inputImage))

            print("inputImage.size = ", inputImage.size)

            if square:
                inputImage = make_square(inputImage, inputImage.mode)
            base_width = inputImage.size[0]
            base_height = inputImage.size[1]
            #inputImage.save("1.jpg")

            # Resize image
            if square:
                inputImageSmaller = inputImage.resize((128, 128), Image.ANTIALIAS)
            else:
                inputImageSmaller = inputImage.resize((96, 128), Image.ANTIALIAS)

            # Delete alpha channel
            print("inputImage.mode ==", inputImageSmaller.mode)
            if inputImageSmaller.mode == "RGBA":
                inputImageSmaller = inputImageSmaller.convert("RGB")

            # Normalization
            inputImageArray = np.asarray(inputImageSmaller)
            prepimg = inputImageArray/ 255.0
            prepimg = prepimg[np.newaxis, :, :, :]

            # Segmentation
            interpreter = tf.contrib.lite.Interpreter(model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_shape = input_details[0]['shape']
            interpreter.set_tensor(input_details[0]['index'], np.array(prepimg, dtype=np.float32))
            t1 = time.time()
            interpreter.invoke()
            print("elapsedtime =", time.time() - t1)
            outputs = interpreter.get_tensor(output_details[0]['index'])

            # Get a color palette
            palette = [0, 0, 0, 255, 255, 255]

            # Define index_void Back Ground
            index_void = 2

            # View
            output = outputs[0]
            res = np.argmax(output, axis=2)
            if index_void is not None:
                res = np.where(res == index_void, 0, res)
            mask = Image.fromarray(np.uint8(res), mode="P")
            mask.putpalette(palette)
            maskRGB = mask.convert("RGB")
            maskRGB = maskRGB.resize((base_width, base_height))
            mask = mask.resize((base_width, base_height))

            #crop to vertical
            if square:
                mask = crop_center(mask, imageMask.width, imageMask.height)
                maskRGB = crop_center(maskRGB, imageMask.width, imageMask.height)
                inputImage = crop_center(inputImage, imageMask.width, imageMask.height)

            IoU = calculate_accuracy(mask, imagesMasks[idx])

            print("IoU Mask" + str(idx+1) + " : " + str(IoU))

            finalImage = get_only_human(inputImage, maskRGB, backgroundImage)

            #finalImage.show()

            nameToSavedMask = relativePathResults + "mask" + str(index) + "_IoU_" + str(int(IoU*1000)) + ".png"
            finalImage.save(nameToSavedMask)
            index += 1





