package com.example.videosegmentation;

import android.content.Intent;
import android.content.pm.PackageManager;

import android.graphics.Bitmap;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageButton;
import android.widget.TextView;

import com.otaliastudios.cameraview.CameraView;

import java.io.File;
import java.io.FileOutputStream;
import java.util.Date;


public class SegmentationActivity extends AppCompatActivity implements ActivityCompat.OnRequestPermissionsResultCallback, View.OnClickListener  {

    private int PERMISSION_REQUEST_CODE = 3;
    private CameraView cameraView;
    private OverlayView overlayViewMask;
    private ImageButton changeBackgroundButton;
    private ImageButton changeModelButton;
    private ImageButton takePictureButton;
    private TextView performanceText;
    private ImageProcessor imageProcessor;
    private String model;


    private class InitializeModelAsyncTask extends AsyncTask<Void, Void, Boolean> {

        @Override
        protected Boolean doInBackground(Void... voids) {
            while(ModelManager.isProcessing()){
                //Log.d("waiting", "waiting");
            }
            //Log.d("Model", "Initializing but is it processing: " + ModelManager.isProcessing());
            final boolean ret = ModelManager.getInstance().initialize(
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

        //Log.d("LIFECYCLE", "OnCreate");
        cameraView = (CameraView) findViewById(R.id.camera_view);
        overlayViewMask = (OverlayView) findViewById(R.id.overlay_view_mask);
        changeBackgroundButton = (ImageButton) findViewById(R.id.changeBackgroundButton);
        changeModelButton = (ImageButton) findViewById(R.id.changeModelButton);
        takePictureButton = (ImageButton) findViewById(R.id.takePictureButton);
        changeBackgroundButton.setOnClickListener(this);
        changeModelButton.setOnClickListener(this);
        takePictureButton.setOnClickListener(this);
        performanceText = (TextView) findViewById(R.id.performanceText);
    }

    protected void onStart(){
        super.onStart();
        ModelManager.NEW_MODEL = getIntent().getStringExtra("model");
        //Log.d("LIFECYCLE", "onStart");
        //Log.d("isProcessing", "the model is processing: " + modelIsProcessing.toString());
        checkAndRequestCameraPermission();

    }

    protected void onStop(){
        super.onStop();
        //Log.d("LIFECYCLE", "onStop");
    }

    protected void onDestroy(){
        super.onDestroy();
//        ModelManager.getInstance().closeModel();
        //Log.d("LIFECYCLE", "onDestroy");
    }

    protected void onRestart(){
        super.onRestart();
        //ModelManager.NEW_MODEL = getIntent().getStringExtra("model");
        //initModel();

    }


    public void checkAndRequestCameraPermission() {
        //check if we have permissions for the camera
        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED
            || ActivityCompat.checkSelfPermission(this, android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            String [] permissions = new String[] {android.Manifest.permission.CAMERA, android.Manifest.permission.WRITE_EXTERNAL_STORAGE};
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
                        //"\nYuv Conversion: " + yuvConversion+
                        "\nInference: " + inference + "ms");

            }
        });

    }

    public void onClick(View view){

        //Log.d("viewID", Integer.toString(view.getId()));

        if(view.getId() == R.id.changeModelButton){
            changeModelButton.setEnabled(false);
            //stop the camera view so no more frame is processed
            //Log.d("TIME", "Trying to stop camera");
            cameraView.stop();
            //Log.d("TIME", "Camera Stopped");
            //get the next segmentation model for that kind
            ModelManager.next();
            ModelManager.NEW_MODEL = ModelManager.MODEL;

        } else if(view.getId() == R.id.changeBackgroundButton){
            overlayViewMask.nextBackground();
        } else if(view.getId() == R.id.takePictureButton){
            takeScreenshot();
        }

    }

    /*
    The following method is called when the model has finished to process each frame
     */
    public void onImageSegmentationEnd(){
        //If the user has pressed to use a new model
        if(ModelManager.NEW_MODEL.equals(ModelManager.MODEL)){
            //Log.d("TIME", "Init new model");
            initModel();
            cameraView.start();
            runOnUiThread(new Runnable() {

                @Override
                public void run() {
                    changeModelButton.setEnabled(true);
                }
            });
        }
    }

    private void takeScreenshot() {
        //Log.d("SCREENSHOT", "Take a screenshot");
        Date now = new Date();
        android.text.format.DateFormat.format("yyyy-MM-dd_hh:mm:ss", now);

        try {
            // image naming and path  to include sd card  appending name you choose for file
            String mPath = Environment.getExternalStorageDirectory().toString() + "/" + now + ".jpg";

            //Log.d("PathToPic", mPath);
            Bitmap bitmap = overlayViewMask.saveScreen(imageProcessor.getLastFrame());

            File imageFile = new File(mPath);

            FileOutputStream outputStream = new FileOutputStream(imageFile);
            int quality = 100;
            bitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream);
            outputStream.flush();
            outputStream.close();

            openScreenshot(imageFile);
        } catch (Throwable e) {
            // Several error may come out with file handling or DOM
            e.printStackTrace();
        }
    }

    private void openScreenshot(File imageFile) {

        Intent intent = new Intent();
        intent.setFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        intent.setAction(Intent.ACTION_VIEW);
        Uri photoURI = FileProvider.getUriForFile(getApplicationContext(), getApplicationContext().getPackageName() + ".provider", imageFile);
        intent.setDataAndType(photoURI, "image/*");
        startActivity(intent);
    }

}


