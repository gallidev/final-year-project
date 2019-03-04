package com.example.videosegmentation;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class SegmentationModel {

    public static final String DEEPLAB = "deeplab";
    public static final String UNET_PORTRAITS= "unet_portraits";
    public static final String UNET_VOC_HUMAN= "unet_voc_human";
    public static String MODEL = "";
    public static String NEW_MODEL = "unet";
    private static AbstractSegmentation aiModel = null;

    private static final String deeplab_path = "deeplabv3_257_mv_gpu.tflite";

    private static final Map<String, String> UnetModelPaths;
    static {
        Map<String, String> aMap = new HashMap<String,String>();
        aMap.put(UNET_PORTRAITS, "128_portraits_26ep_32ba_quantized_32.tflite");
        aMap.put(UNET_VOC_HUMAN, "semanticsegmentation_frozen_person_quantized_32.tflite");
        UnetModelPaths = Collections.unmodifiableMap(aMap);
    }


    //this makes sure we always get one and only one instance of the class even
    // with multi-threads
    public synchronized static AbstractSegmentation getInstance() {

        if (NEW_MODEL.equals("") && aiModel != null) {
            return aiModel;
        }

        if (NEW_MODEL.equals(UNET_PORTRAITS)) {
            aiModel = new UnetPortraits(UnetModelPaths.get(NEW_MODEL));
            MODEL = NEW_MODEL;
            NEW_MODEL ="";
        } else if (NEW_MODEL.equals(DEEPLAB)) {
            aiModel = new Deeplab(deeplab_path);
            MODEL = NEW_MODEL;
            NEW_MODEL = "";
        } else if (NEW_MODEL.equals(UNET_VOC_HUMAN)){
            aiModel = new UnetVocHuman(UnetModelPaths.get(NEW_MODEL));
            MODEL = NEW_MODEL;
            NEW_MODEL = "";
        }


        return aiModel;
    }




}
