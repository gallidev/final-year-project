package com.example.videosegmentation;

import android.content.pm.PackageManager;

import android.os.AsyncTask;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import com.dailystudio.development.Logger;
import com.otaliastudios.cameraview.CameraView;


public class SegmentationActivity extends AppCompatActivity implements ActivityCompat.OnRequestPermissionsResultCallback  {

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
            final boolean ret = SegmentationModel.getInstance().initialize(
                    getApplicationContext());

            Log.d("Model", "Initialized model");

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

        SegmentationModel.NEW_MODEL = getIntent().getStringExtra("model");

        cameraView = (CameraView) findViewById(R.id.camera_view);
        overlayViewMask = (OverlayView) findViewById(R.id.overlay_view_mask);
        performanceText = (TextView) findViewById(R.id.performanceText);
        //overlayViewCropped = (OverlayView) findViewById(R.id.overlay_view_cropped);
        checkAndRequestCameraPermission();
    }

    protected void onRestart(){
        super.onRestart();
        SegmentationModel.NEW_MODEL = getIntent().getStringExtra("model");
        initModel();

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

        Log.d("Starting Processing", "Starting processing");

        imageProcessor.startProcessing();
        //faceProcessor = new FaceProcessor(cameraView, overlayView);
        //faceProcessor.startProcessing();
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

    public void showPerformance(final String yuvConversion, final String imageManipulation, final String inference){

        runOnUiThread(new Runnable() {

            @Override
            public void run() {

                performanceText.setText("Yuv Conversion: " + yuvConversion +
                        "\nImage Manipulation: " + imageManipulation +
                        "\nInference: " + inference);

            }
        });

    }

}


