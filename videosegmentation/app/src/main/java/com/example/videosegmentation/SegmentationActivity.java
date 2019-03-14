package com.example.videosegmentation;

import android.content.pm.PackageManager;

import android.os.AsyncTask;
import android.os.Handler;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import com.dailystudio.development.Logger;
import com.otaliastudios.cameraview.CameraView;


public class SegmentationActivity extends AppCompatActivity implements ActivityCompat.OnRequestPermissionsResultCallback, View.OnClickListener  {

    private int PERMISSION_REQUEST_CODE = 3;
    private CameraView cameraView;
    private OverlayView overlayViewMask;
    //private OverlayView overlayViewCropped;
    private TextView performanceText;
    private ImageProcessor imageProcessor;
    private String model;


    private class InitializeModelAsyncTask extends AsyncTask<Void, Void, Boolean> {

        @Override
        protected Boolean doInBackground(Void... voids) {
            while(SegmentationModel.isProcessing()){
                //Log.d("waiting", "waiting");
            }
            //Log.d("Model", "Initializing but is it processing: " + SegmentationModel.isProcessing());
            final boolean ret = SegmentationModel.getInstance().initialize(
                    getApplicationContext());

            //Log.d("Model", "Initialized model");
            //cameraView.start();
            return ret;
        }

    }

    private void initModel() {
        new InitializeModelAsyncTask().execute((Void)null);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Make this activity full screen
        //remove title
        setContentView(R.layout.activity_segmentation_full);

        Log.d("LIFECYCLE", "OnCreate");
        cameraView = (CameraView) findViewById(R.id.camera_view);
        overlayViewMask = (OverlayView) findViewById(R.id.overlay_view_mask);
        overlayViewMask.setOnClickListener(this);
        performanceText = (TextView) findViewById(R.id.performanceText);
        //overlayViewCropped = (OverlayView) findViewById(R.id.overlay_view_cropped);

    }

    protected void onStart(){
        super.onStart();
        SegmentationModel.NEW_MODEL = getIntent().getStringExtra("model");
        Log.d("LIFECYCLE", "onStart");
        //Log.d("isProcessing", "the model is processing: " + modelIsProcessing.toString());
        checkAndRequestCameraPermission();

    }

    protected void onStop(){
        super.onStop();
        cameraView.stop();
        cameraView.destroy();
    }

    protected void onRestart(){
        super.onRestart();
        //SegmentationModel.NEW_MODEL = getIntent().getStringExtra("model");
        //initModel();

    }


    public void checkAndRequestCameraPermission() {
        //check if we have permissions for the camera
        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            String [] permissions = new String[] {android.Manifest.permission.CAMERA};
            //if not request permissions
            ActivityCompat.requestPermissions(this, permissions, PERMISSION_REQUEST_CODE );
        } else {
            startFaceProcessor();
        }
    }

    public void startFaceProcessor() {
        getLifecycle().addObserver(new MainActivityLifecycleObserver(cameraView));

        initModel();

        imageProcessor = new ImageProcessor(cameraView, overlayViewMask, this, getApplicationContext());

        //Log.d("Starting Processing", "Starting processing");

        imageProcessor.startProcessing();

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String [] permissions, int [] grantResults) {
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (android.Manifest.permission.CAMERA == permissions[0] &&
                    grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startFaceProcessor();
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    public void showPerformance(final String modelPath, final String yuvConversion, final String inference){

        runOnUiThread(new Runnable() {

            @Override
            public void run() {
                performanceText.setText("Model: " + modelPath +
                        "\nYuv Conversion: " + yuvConversion+
                        "\nInference: " + inference);

            }
        });

    }

    public void onClick(View view){

        if(view.getId() == R.id.overlay_view_mask){
            //stop the camera view so no more frame is processed
            cameraView.stop();

            //get the next segmentation model for that kind
            SegmentationModel.next();
            SegmentationModel.NEW_MODEL = SegmentationModel.MODEL;

        }
    }

    /*
    The following method is called when the model has finished to process each frame
     */
    public void onImageSegmentationEnd(){
        //If the user has tapped prepare the new model
        if(SegmentationModel.NEW_MODEL.equals(SegmentationModel.MODEL)){
            Log.d("TIME", "Init new model");
            initModel();
            cameraView.start();
        }
    }

}


