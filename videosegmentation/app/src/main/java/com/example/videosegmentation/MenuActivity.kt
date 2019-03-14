package com.example.videosegmentation

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.View
import com.example.videosegmentation.SegmentationModel.*

class MenuActivity : Activity() {



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_menu)
    }

    fun onDeeplabClick(view: View){
        val intent = Intent(this, SegmentationActivity::class.java)
        intent.putExtra("model", DEEPLAB)
        startActivity(intent)
    }

    fun onUnetVocClick(view: View){
        val intent = Intent(this, SegmentationActivity::class.java)
        intent.putExtra("model", UNET_VOC_HUMAN)
        startActivity(intent)

    }

    fun onUnetPortraitsClick(view: View){
        val intent = Intent(this, SegmentationActivity::class.java)
        intent.putExtra("model", UNET_PORTRAITS)
        startActivity(intent)

    }

    fun onUnetPortraitsSmallerClick(view: View){
        val intent = Intent(this, SegmentationActivity::class.java)
        intent.putExtra("model", UNET_PORTRAITS_SMALLER)
        startActivity(intent)

    }



}
