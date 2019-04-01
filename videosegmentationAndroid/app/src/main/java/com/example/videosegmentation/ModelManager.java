package com.example.videosegmentation;

import android.util.Log;

public class ModelManager {

    public static final String DEEPLAB = "deeplab";
    public static final String UNET_PORTRAITS = "unet_portraits";
    public static final String UNET_VOC_HUMAN = "unet_voc_human";
    public static final String UNET_PORTRAITS_SMALLER = "unet_portraits_smaller";
    public static String MODEL = "";
    public static String NEW_MODEL = "unet";
    private static AbstractSegmentation aiModel = null;
    private static boolean isProcessing = false;

    private static final Model[] unetPortrait = {
            new Model("1_model_20e_128_quantized.tflite", "Standard 20e"),
            new Model("2_model_26e_128_quantized.tflite", "Half Conv2D 26e"),
            new Model("3_model_26e_128_quantized.tflite", "Half Conv2D + Big strides 26e")
    };
    private static final Model[] unetPortraitSmaller = {
            new Model("4_model_26e_96_128_quantized.tflite", "Half Conv2D 26e"),
            new Model("4_model_12e_96_128_quantized.tflite", "Half Conv2D 12e"),
            new Model("4_model_12e_96_128_aug_quantized.tflite", "Half Conv2D Aug 12e"),
            new Model("5_model_18e_96_128_aug_quantized.tflite", "Half Conv2D Aug + Big strides 18e"),
            new Model("5_model_32e_96_128_aug_quantized.tflite", "Half Conv2D Aug + Big strides 32e")

    };

    private static final Model deeplab =  new Model("deeplabv3_257_mv_gpu.tflite", "DeeplabV3+");
    private static final Model unetVOC = new Model("semanticsegmentation_frozen_person_quantized_32.tflite", "UNet standard");  ;

    private static int modelIndex = 0;


    //this makes sure we always get one and only one instance of the class even
    // with multi-threads
    public synchronized static AbstractSegmentation getInstance() {

        if (NEW_MODEL.equals("") && aiModel != null) {
            return aiModel;
        }

        if(aiModel != null && aiModel.isInitialized()) {
            aiModel.closeModel();
        }

        if (NEW_MODEL.equals(UNET_PORTRAITS)) {
            if(modelIndex % unetPortrait.length == 0){
                modelIndex = 0;
            }
            aiModel = new UnetPortraits(unetPortrait[modelIndex].path);
            //Log.d("new model", "model index " + modelIndex);
        } else if (NEW_MODEL.equals(DEEPLAB)) {
            aiModel = new Deeplab(deeplab.path);
        } else if (NEW_MODEL.equals(UNET_VOC_HUMAN)){
            aiModel = new UnetVocHuman(unetVOC.path);
        } else if (NEW_MODEL.equals(UNET_PORTRAITS_SMALLER)){
            if(modelIndex % unetPortraitSmaller.length == 0){
                modelIndex = 0;
            }
            aiModel = new UnetPortraitsSmaller(unetPortraitSmaller[modelIndex].path);
            //Log.d("new model", "model index " + modelIndex);
        }

        MODEL = NEW_MODEL;
        NEW_MODEL ="";

        return aiModel;
    }


    //start the next model
    public synchronized static void next() {
        increaseModelIndex();
    }


    private synchronized static void increaseModelIndex(){
        switch (MODEL){
            case UNET_PORTRAITS:
            {
                modelIndex ++;
                if(modelIndex % unetPortrait.length == 0){
                    modelIndex = 0;
                }
                break;
            }
            case UNET_PORTRAITS_SMALLER:
            {
                modelIndex ++;
                //Log.d("new model", "model index " + modelIndex);
                if(modelIndex % unetPortraitSmaller.length == 0){
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

    public synchronized static String getModelName(){
        if (MODEL.equals(UNET_PORTRAITS)) {
            return unetPortrait[modelIndex].name;
        } else if (MODEL.equals(DEEPLAB)) {
            return deeplab.name;
        } else if (MODEL.equals(UNET_VOC_HUMAN)){
            return unetVOC.name;
        } else if (MODEL.equals(UNET_PORTRAITS_SMALLER)){
            return unetPortraitSmaller[modelIndex].name;
        }
        return "";
    }

}
