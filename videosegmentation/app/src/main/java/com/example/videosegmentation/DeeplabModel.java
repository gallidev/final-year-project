package com.example.videosegmentation;

public class DeeplabModel {

    private final static Boolean USE_GPU = true;

    private static DeeplabGPU sInterface = null;

    //this makes sure we always get one and only one instance of the class even
    // with multi-threads
    public synchronized static DeeplabGPU getInstance() {
        if (sInterface != null) {
            return sInterface;
        }

        if (USE_GPU) {
            sInterface = new DeeplabGPU();
        } else {
            //sInterface = new DeeplabMobile();
        }

        return sInterface;
    }

}
