import sys, time
import numpy as np
import tensorflow as tf
from PIL import Image

def make_square(im, imageMode, init_size=(256,256), fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(init_size[0], x, y)
    #check if the image is a png it won't accept the color black in 4 values
    if(imageMode == 'L' or imageMode == 'P'):
        fill_color = 0
    new_im = Image.new(imageMode, (size, size), fill_color)
    new_im.paste(im, ((size - x) // 2, (size - y) // 2))
    return new_im


if __name__ == '__main__':

    imageFirst = Image.open("images/test1.jpeg")
    imageSecond = Image.open("images/test2.jpeg")
    imageThird = Image.open("images/test3.jpeg")
    #seg_image = Image.open("data_set/portraits/masks_png/00001.png")
    print("image.size = ", imageFirst.size)
    
    image = make_square(imageFirst, imageFirst.mode)
    image.save("1.jpg")
    base_width  = image.size[0]
    base_height = image.size[1]
    # resize image
    image = image.resize((256, 256), Image.ANTIALIAS)

    image.show()

    # delete alpha channel
    print("image.mode ==", image.mode)
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # normalization
    image = np.asarray(image)
    prepimg = image / 255.0

    # 1 Channel -> 3 Channels convert
    if prepimg.ndim < 3:
        prepimg = prepimg[:, :, np.newaxis]
        prepimg = np.insert(prepimg, 1, prepimg[:,:,0], axis=2)
        prepimg = np.insert(prepimg, 2, prepimg[:,:,0], axis=2)

    # Read .pb file
    #with tf.gfile.FastGFile("model/Test_22_Selfies_256_model/semanticsegmentation_frozen_person.pb", "rb") as f:
    with tf.gfile.FastGFile("models/semanticsegmentation_frozen_person_32.pb", "rb") as f:
        graphdef = tf.GraphDef()
        graphdef.ParseFromString(f.read())
        _ = tf.import_graph_def(graphdef, name="")
    sess = tf.Session()

    # Segmentation
    t1 = time.time()
    outputs = sess.run("output/BiasAdd:0", {"input:0":[prepimg]})
    print("elapsedtime =", time.time() - t1)

    # Get a color palette
    #palette = seg_image.getpalette()
    #if palette is None:
    palette = [0,0,0, 255,0,0]

    # Define index_void (len(DataSet.CATEGORY)-1)
    index_void = 2

    # View
    output = outputs[0]
    res = np.argmax(output, axis=2)
    if index_void is not None:
        res = np.where(res == index_void, 0, res)
    image = Image.fromarray(np.uint8(res), mode="P")
    image.putpalette(palette)
    image = image.convert("RGB")

    image = image.resize((base_width, base_height))

    image.save("2.jpg")



