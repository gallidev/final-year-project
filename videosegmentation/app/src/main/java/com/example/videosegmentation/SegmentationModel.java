package com.example.videosegmentation;

import android.util.Log;

public class SegmentationModel {

    public static final String DEEPLAB = "deeplab";
    public static final String UNET_PORTRAITS = "unet_portraits";
    public static final String UNET_VOC_HUMAN = "unet_voc_human";
    public static final String UNET_PORTRAITS_SMALLER = "unet_portraits_smaller";
    public static String MODEL = "";
    public static String NEW_MODEL = "unet";
    private static AbstractSegmentation aiModel = null;
    private static boolean isProcessing = false;

    private static final String[] unetPortraitsPaths = {
            "1_model_20e_128_quantized.tflite",
            "2_model_26e_128_quantized.tflite",
            "3_model_26e_128_quantized.tflite"
    };
    private static final String deeplabPath = "deeplabv3_257_mv_gpu.tflite";
    private static final String unetportraitSmaller = "4_model_26e_96_128_quantized.tflite";
    private static final String unetVOCPath = "semanticsegmentation_frozen_person_quantized_32.tflite";

    private static int modelIndex = 0;


    //this makes sure we always get one and only one instance of the class even
    // with multi-threads
    public synchronized static AbstractSegmentation getInstance() {

        if (NEW_MODEL.equals("") && aiModel != null) {
            return aiModel;
        }

        if(aiModel != null) {
            aiModel.closeModel();
        }

        if (NEW_MODEL.equals(UNET_PORTRAITS)) {
            aiModel = new UnetPortraits(unetPortraitsPaths[modelIndex]);
            Log.d("new model", "model index " + modelIndex);
        } else if (NEW_MODEL.equals(DEEPLAB)) {
            aiModel = new Deeplab(deeplabPath);
        } else if (NEW_MODEL.equals(UNET_VOC_HUMAN)){
            aiModel = new UnetVocHuman(unetVOCPath);
        } else if (NEW_MODEL.equals(UNET_PORTRAITS_SMALLER)){
            aiModel = new UnetPortraitsSmaller(unetportraitSmaller);
        }

        MODEL = NEW_MODEL;
        NEW_MODEL ="";

        return aiModel;
    }


    //start the next model
    public synchronized static void next() {

        if (MODEL.equals(UNET_PORTRAITS)) {
            increaseModelIndex();
        }
    }


    private synchronized static void increaseModelIndex(){
        switch (MODEL){
            case UNET_PORTRAITS:
            {
                modelIndex ++;
                if(modelIndex % unetPortraitsPaths.length == 0){
                    modelIndex = 0;
                }
                break;
            }
        }

    }

    public synchronized static void setIsProcessing(boolean processing){
      isProcessing = processing;
    }

    public synchronized static boolean isProcessing(){
       return isProcessing;
    }

    public synchronized static String getModelPath(){
        if (MODEL.equals(UNET_PORTRAITS)) {
            return unetPortraitsPaths[modelIndex];
        } else if (MODEL.equals(DEEPLAB)) {
            return deeplabPath;
        } else if (MODEL.equals(UNET_VOC_HUMAN)){
            return unetVOCPath;
        } else if (MODEL.equals(UNET_PORTRAITS_SMALLER)){
            return unetportraitSmaller;
        }
        return "";
    }

}
