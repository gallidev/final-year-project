package com.example.videosegmentation;

public class SegmentationModel {

    private final static Boolean USE_GPU = false;

    private static DeeplabGPU sInterface = null;
    private static UnetCPU aiModel = null;

    //this makes sure we always get one and only one instance of the class even
    // with multi-threads
    public synchronized static DeeplabGPU getInstance() {
        if (sInterface != null) {
            return sInterface;
        }

        if (USE_GPU) {
            sInterface = new DeeplabGPU();
        }

        return sInterface;
    }

    //this makes sure we always get one and only one instance of the class even
    // with multi-threads
    public synchronized static UnetCPU getUnetInstance() {
        if (aiModel != null) {
            return aiModel;
        }

        if (!USE_GPU) {
            aiModel = new UnetCPU();
        }

        return aiModel;
    }


}
