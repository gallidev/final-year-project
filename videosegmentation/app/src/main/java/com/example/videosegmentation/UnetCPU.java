package com.example.videosegmentation;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.text.TextUtils;
import android.util.Log;

import com.dailystudio.app.utils.ArrayUtils;
import com.dailystudio.app.utils.BitmapUtils;
import com.dailystudio.development.Logger;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Random;

public class UnetCPU {

    private final static String MODEL_PATH = "128_portraits_26ep_32ba_quantized_32.tflite";

    //the model input can only be 128x128 in terms of size
    private final static Integer INPUT_SIZE = 128;
    private final static int NUM_CLASSES = 2;

    private volatile Interpreter sTfInterpreter = null;

    private int[][] mSegmentBits;
    private int[] mSegmentColors;

    private final static Random RANDOM = new Random(System.currentTimeMillis());

    public boolean initialize(Context context) {
        if (context == null) {
            return false;
        }

        MappedByteBuffer buffer = loadModelFile(context, MODEL_PATH);
        if (buffer == null) {
            return false;
        }

        Interpreter.Options options = new Interpreter.Options();

//        GpuDelegate delegate = new GpuDelegate();
//        options.addDelegate(delegate);

        sTfInterpreter = new Interpreter(buffer, options);

        debugInputs(sTfInterpreter);
        debugOutputs(sTfInterpreter);


        mSegmentBits = new int[INPUT_SIZE][INPUT_SIZE];
        mSegmentColors = new int[NUM_CLASSES];
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (i == 0) {
                mSegmentColors[i] = Color.TRANSPARENT;
            } else {
                mSegmentColors[i] = Color.rgb(
                        (int)(255 * RANDOM.nextFloat()),
                        (int)(255 * RANDOM.nextFloat()),
                        (int)(255 * RANDOM.nextFloat()));
            }
        }

        return (sTfInterpreter != null);
    }


    public boolean isInitialized() {
        return (sTfInterpreter != null);
    }

    static public float getInputSize() {
        return INPUT_SIZE;
    }

    public Bitmap segment(Bitmap bitmap) {
        if (sTfInterpreter == null) {
            Log.w("model", "tf model is NOT initialized.");
            return null;
        }

        if (bitmap == null) {
            return null;
        }

        int w = bitmap.getWidth();
        int h = bitmap.getHeight();
        Logger.debug("bitmap: %d x %d,", w, h);

        if (w > INPUT_SIZE || h > INPUT_SIZE) {
           Logger.warn("invalid bitmap size: %d x %d [should be: %d x %d]",
                    w, h,
                    INPUT_SIZE, INPUT_SIZE);

            return null;
        }

        // the size needs to be smaller than what the model can run in this example
        // it also needs to be a square 128x128
        // if the picture is a vertical one it will create black bands on the sides
        if (w < INPUT_SIZE || h < INPUT_SIZE) {
            bitmap = BitmapUtils.extendBitmap(
                    bitmap, INPUT_SIZE, INPUT_SIZE, Color.BLACK);

            w = bitmap.getWidth();
            h = bitmap.getHeight();
            Logger.debug("extend bitmap: %d x %d,", w, h);
        }


        int[] mIntValues = new int[w * h];
        //I had to put 3 here probably because of the colors available after [1][w][h][3]
        float[][][][] mInput = new float[1][w][h][3];
        //float[][][][] mOutputs = new float[1][w][h][3];
        float[][][][] mOutputs = new float[1][w][h][2];

        bitmap.getPixels(mIntValues, 0, w, 0, 0, w, h);

        //normalise the values of the image
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                if (pixel >= mIntValues.length) {
                    break;
                }

                final int val = mIntValues[pixel++];
                //find more info here on how to get the colors out
                //https://stackoverflow.com/questions/5669501/how-do-you-get-the-rgb-values-from-a-bitmap-on-an-android-device

                mInput[0][i][j][0] = ((val >> 16) & 0xFF)/ 255f;
                mInput[0][i][j][1] = ((val >> 8) & 0xFF)/ 255f;
                mInput[0][i][j][2] = (val & 0xFF)/ 255f;
            }
        }

        final long start = System.currentTimeMillis();

        sTfInterpreter.run(mInput, mOutputs);

        //to get out the segmentation mask from mOutputs we need to see when the second
        // float of each pixel is the highest number of the 3

        final long end = System.currentTimeMillis();
        Log.d("TIME", "CORE inference " + java.lang.Long.toString(end - start));


        Bitmap output = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (mOutputs[0][y][x][1] > mOutputs[0][y][x][0]
                        //&& mOutputs[0][y][x][1] > mOutputs[0][y][x][2])
                    )
                {
                    output.setPixel(x, y, mSegmentColors[1]);
                } else {
                    output.setPixel(x, y, mSegmentColors[0]);
                }

            }
        }


        return output;
    }

    private void fillZeroes(int[][] array) {
        if (array == null) {
            return;
        }

        int r;
        for (r = 0; r < array.length; r++) {
            Arrays.fill(array[r], 0);
        }
    }

    private static void debugInputs(Interpreter interpreter) {
        if (interpreter == null) {
            return;
        }

        final int numOfInputs = interpreter.getInputTensorCount();
        Logger.debug("[TF-LITE-MODEL] input tensors: [%d]",numOfInputs);

        for (int i = 0; i < numOfInputs; i++) {
            Tensor t = interpreter.getInputTensor(i);
            Logger.debug("[TF-LITE-MODEL] input tensor[%d[: shape[%s]",
                    i,
                    ArrayUtils.intArrayToString(t.shape()));
        }
    }

    private static void debugOutputs(Interpreter interpreter) {
        if (interpreter == null) {
            return;
        }

        final int numOfOutputs = interpreter.getOutputTensorCount();
        Logger.debug("[TF-LITE-MODEL] output tensors: [%d]",numOfOutputs);

        for (int i = 0; i < numOfOutputs; i++) {
            Tensor t = interpreter.getOutputTensor(i);
            Logger.debug("[TF-LITE-MODEL] output tensor[%d[: shape[%s]",
                    i,
                    ArrayUtils.intArrayToString(t.shape()));
        }
    }

    private static MappedByteBuffer loadModelFile(Context context, String modelFile) {
        if (context == null
                || TextUtils.isEmpty(modelFile)) {
            return null;
        }

        MappedByteBuffer buffer = null;

        try {
            AssetFileDescriptor df = context.getAssets().openFd(modelFile);

            FileInputStream inputStream = new FileInputStream(df.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = df.getStartOffset();
            long declaredLength = df.getDeclaredLength();

            buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            Logger.debug("load tflite model from [%s] failed: %s",
                    modelFile,
                    e.toString());

            buffer = null;
        }

        return buffer;
    }

}