package com.example.videosegmentation;

import android.content.pm.PackageManager;

import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.WindowManager;

import com.otaliastudios.cameraview.CameraView;


public class MainActivity extends AppCompatActivity implements ActivityCompat.OnRequestPermissionsResultCallback  {

    private int PERMISSION_REQUEST_CODE = 3;
    private CameraView cameraView;
    private OverlayView overlayView;
    private FaceProcessor faceProcessor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Make this activity full screen
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_main);

        cameraView = (CameraView) findViewById(R.id.camera_view);
        overlayView = (OverlayView) findViewById(R.id.overlay_view);

        checkAndRequestCameraPermission();
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

        faceProcessor = new FaceProcessor(cameraView, overlayView);
        faceProcessor.startProcessing();
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

}


