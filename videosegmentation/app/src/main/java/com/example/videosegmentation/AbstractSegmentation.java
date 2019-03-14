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

import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Random;

public abstract class AbstractSegmentation {

    protected String model_path;
    protected boolean isProcessing;

    protected Delegate gpuDelegate = null;

    protected volatile Interpreter sTfInterpreter = null;

    /** Options for configuring the Interpreter. */
    protected final Interpreter.Options tfliteOptions = new Interpreter.Options();

    protected int[][] mSegmentBits;
    protected int[] mSegmentColors;

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    protected ByteBuffer imgData = null;

    protected final static Random RANDOM = new Random(System.currentTimeMillis());

    public abstract boolean initialize(Context context);

    public boolean isInitialized() {
        return (sTfInterpreter != null);
    }

    public abstract Bitmap segment(Bitmap bitmap);

    protected void fillZeroes(int[][] array) {
        if (array == null) {
            return;
        }

        int r;
        for (r = 0; r < array.length; r++) {
            Arrays.fill(array[r], 0);
        }
    }

    protected void debugInputs(Interpreter interpreter) {
        if (interpreter == null) {
            return;
        }

        final int numOfInputs = interpreter.getInputTensorCount();
        //Log.d("TF-LITE-MODEL", "input tensors: " + numOfInputs);

        for (int i = 0; i < numOfInputs; i++) {
            Tensor t = interpreter.getInputTensor(i);
            //Log.d("TF-LITE-MODEL", "input tensor " + i
            //        + "shape" +ArrayUtils.intArrayToString(t.shape()));
        }
    }

    protected void debugOutputs(Interpreter interpreter) {
        if (interpreter == null) {
            return;
        }

        final int numOfOutputs = interpreter.getOutputTensorCount();
        //Log.d("TF-LITE-MODEL", "output tensors: " + numOfOutputs);

        for (int i = 0; i < numOfOutputs; i++) {
            Tensor t = interpreter.getOutputTensor(i);
            //Log.d("TF-LITE-MODEL", "output tensor " + i
            //       + "shape" +ArrayUtils.intArrayToString(t.shape()));
        }
    }

    protected MappedByteBuffer loadModelFile(Context context, String modelFile) {
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
            //Logger.debug("load tflite model from [%s] failed: %s",
            //        modelFile,
            //        e.toString());

            buffer = null;
        }

        return buffer;
    }

    protected void addPixelValue(int pixelValue) {
        imgData.putFloat(((pixelValue >> 16) & 0xFF) / 255.f);
        imgData.putFloat(((pixelValue >> 8) & 0xFF) / 255.f);
        imgData.putFloat((pixelValue & 0xFF) / 255.f);
    }

    public void closeModel(){
        if(sTfInterpreter != null){
            sTfInterpreter.close();
        }

    }

    public void setProcessing(boolean isProcessing){
        this.isProcessing = isProcessing;
    }

    public boolean isProcessing(){
        return this.isProcessing;
    }

}
