package com.example.videosegmentation;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;

import com.dailystudio.app.utils.BitmapUtils;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.experimental.GpuDelegate;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;

public class UnetPortraits extends AbstractSegmentation{


    //the model input can only be 128x128 in terms of size
    private final static Integer INPUT_SIZE = 128;
    private final static int NUM_CLASSES = 2;

    public UnetPortraits(String model_path){
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

        tfliteOptions.setNumThreads(1);
/*
        if(GpuDelegateHelper.isGpuDelegateAvailable()){
            Log.d("GPU", "initializing with GPU delegate");
            gpuDelegate = GpuDelegateHelper.createGpuDelegate();
            options.addDelegate(gpuDelegate);
        }else{
            Log.d("GPU", "initializing without GPU delegate");
        }
*/
        sTfInterpreter = new Interpreter(buffer, tfliteOptions);

        //debugInputs(sTfInterpreter);
        //debugOutputs(sTfInterpreter);

        mSegmentBits = new int[INPUT_SIZE][INPUT_SIZE];
        mSegmentColors = new int[NUM_CLASSES];
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (i == 0) {
                mSegmentColors[i] = Color.TRANSPARENT;
            } else {
                mSegmentColors[i] = Color.rgb(255,255,255);
            }
        }

        imgData =
                ByteBuffer.allocateDirect(
                                INPUT_SIZE
                                * INPUT_SIZE
                                * 3
                                * 4);
        imgData.order(ByteOrder.nativeOrder());

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
           //Log.d("invalid bitmap size: %d x %d [should be: %d x %d]",
           //         w, h,
            //        INPUT_SIZE, INPUT_SIZE);

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
            //Logger.debug("extend bitmap: %d x %d,", w, h);
        }


        int[] mIntValues = new int[w * h];

        float[][][][] mOutputs = new float[1][w][h][2];

        bitmap.getPixels(mIntValues, 0, w, 0, 0, w, h);

        imgData.rewind();

        //normalise the values of the image
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                if (pixel >= mIntValues.length) {
                    break;
                }
                final int val = mIntValues[pixel++];
                addPixelValue(val);
            }
        }

        final long start = System.currentTimeMillis();
        sTfInterpreter.run(imgData, mOutputs);
        //to get out the segmentation mask from mOutputs we need to see when the second
        // float of each pixel is the highest number of the 3

        final long end = System.currentTimeMillis();
        //Log.d("TIME", "CORE inference " + java.lang.Long.toString(end - start));


        Bitmap output = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (mOutputs[0][y][x][1] < mOutputs[0][y][x][0]
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


}