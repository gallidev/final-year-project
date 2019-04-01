package com.example.videosegmentation

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.View
import com.example.videosegmentation.ModelManager.*

class MenuActivity : Activity() {

    //to prevent multiple activities started
    var intentCalled = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_menu)
        intentCalled = false
    }

    override fun onStart() {
        super.onStart()
        intentCalled = false
    }

    fun onDeeplabClick(view: View){
        if(!intentCalled) {
            val intent = Intent(this, SegmentationActivity::class.java)
            intent.putExtra("model", DEEPLAB)
            startActivity(intent)
            intentCalled = true
        }
    }

    fun onUnetVocClick(view: View){
        if(!intentCalled) {
            val intent = Intent(this, SegmentationActivity::class.java)
            intent.putExtra("model", UNET_VOC_HUMAN)
            startActivity(intent)
            intentCalled = true
        }

    }

    fun onUnetPortraitsClick(view: View) {
        if (!intentCalled) {
            val intent = Intent(this, SegmentationActivity::class.java)
            intent.putExtra("model", UNET_PORTRAITS)
            startActivity(intent)
            intentCalled = true
        }

    }

    fun onUnetPortraitsSmallerClick(view: View){
        if(!intentCalled) {
            val intent = Intent(this, SegmentationActivity::class.java)
            intent.putExtra("model", UNET_PORTRAITS_SMALLER)
            startActivity(intent)
            intentCalled = true
        }

    }




}
