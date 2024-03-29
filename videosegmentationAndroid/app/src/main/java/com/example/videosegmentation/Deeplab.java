package com.example.videosegmentation;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;

import com.dailystudio.app.utils.BitmapUtils;

import org.tensorflow.lite.Interpreter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.Random;

public class Deeplab extends AbstractSegmentation{

    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;
    //the model input can only be 257x257 in terms of size
    private final static Integer INPUT_SIZE = 257;
    private final static int NUM_CLASSES = 21;
    private final static int HUMAN_CLASS_INDEX = 15;

    private volatile Interpreter sTfInterpreter = null;

    private ByteBuffer mImageData;

    private final static Random RANDOM = new Random(System.currentTimeMillis());

    public Deeplab(String model_path){
        super();
        this.model_path = model_path;
    }

    public boolean initialize(Context context) {
        if (context == null) {
            return false;
        }

        MappedByteBuffer buffer = loadModelFile(context, model_path);
        if (buffer == null) {
            return false;
        }

        sTfInterpreter = new Interpreter(buffer, tfliteOptions);

        debugInputs(sTfInterpreter);
        debugOutputs(sTfInterpreter);

        mImageData =
                ByteBuffer.allocateDirect(
                        1 * INPUT_SIZE * INPUT_SIZE * 3 * 4);
        mImageData.order(ByteOrder.nativeOrder());

        mSegmentBits = new int[INPUT_SIZE][INPUT_SIZE];
        mSegmentColors = new int[NUM_CLASSES];


        for (int i = 0; i < NUM_CLASSES; i++) {
                mSegmentColors[i] = Color.rgb(255,255,255);

        }

        mSegmentColors[HUMAN_CLASS_INDEX] = Color.TRANSPARENT;


        return (sTfInterpreter != null);
    }

    static public float getInputSize() {
        return INPUT_SIZE;
    }

    public Bitmap segment(Bitmap bitmap) {
        if (sTfInterpreter == null) {
            //Log.w("model", "tf model is NOT initialized.");
            return null;
        }

        if (bitmap == null) {
            return null;
        }

        int w = bitmap.getWidth();
        int h = bitmap.getHeight();


        if (w > INPUT_SIZE || h > INPUT_SIZE) {

            return null;
        }

        // the size needs to be smaller than what the model can run in this example
        // it also needs to be a square 257x257
        // if the picture is a vertical one it will create black bands on the sides
        if (w < INPUT_SIZE || h < INPUT_SIZE) {
            bitmap = BitmapUtils.extendBitmap(
                    bitmap, INPUT_SIZE, INPUT_SIZE, Color.BLACK);

            w = bitmap.getWidth();
            h = bitmap.getHeight();
        }

        mImageData.rewind();

        int[] mIntValues = new int[w * h];
        float[][][][] mOutputs = new float[1][w][h][21];

        bitmap.getPixels(mIntValues, 0, w, 0, 0, w, h);

        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                if (pixel >= mIntValues.length) {
                    break;
                }

                final int val = mIntValues[pixel++];
                mImageData.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                mImageData.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                mImageData.putFloat(((val & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }

        final long start = System.currentTimeMillis();

        sTfInterpreter.run(mImageData, mOutputs);

        final long end = System.currentTimeMillis();

        Bitmap output = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);

        fillZeroes(mSegmentBits);
        float maxVal = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                mSegmentBits[x][y] = 0;

                for (int c = 0; c < NUM_CLASSES; c++) {
                    if (c == 0 || mOutputs[0][y][x][c] > maxVal) {
                        maxVal = mOutputs[0][y][x][c];
                        mSegmentBits[x][y] = c;
                    }
                }

                output.setPixel(x, y, mSegmentColors[mSegmentBits[x][y]]);
            }
        }


        return output;
    }

}