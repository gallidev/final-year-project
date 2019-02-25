package com.example.videosegmentation

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.util.Log
import android.view.View

/**
 * A overlay view that draws thug life glasses and cigarette bitmaps on top of a detected face
 *
 * Created by Qichuan on 21/6/18.
 */
class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {


    var mask: Bitmap? = null
    var oldMask: Bitmap? = null

    // The preview width
    var previewWidth: Int? = null

    // The preview height
    var previewHeight: Int? = null

    private var widthScaleFactor = 1.0f
    private var heightScaleFactor = 1.0f

    // The glasses bitmap
    private val glassesBitmap: Bitmap = BitmapFactory.decodeResource(resources, R.drawable.glasses)

    // The cigarette bitmap
    private val cigaretteBitmap: Bitmap = BitmapFactory.decodeResource(resources, R.drawable.cigarette)

    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)

        // Create local variables here so they cannot not be changed anywhere else
        // val face = face
        val maskFixed = mask

        Log.d("draw", "checking if the mask or old mask are not null")
        if(maskFixed != null && canvas != null){

            val maskRect = Rect(
                    0,
                    0,
                    canvas.width,
                    canvas.height)


            Log.d("drawing Mask:", maskFixed.getPixel(60,50).toString() );
            canvas.drawBitmap(maskFixed, null, maskRect, null)
        }

        Log.d("draw", "checking if the mask or old mask are not null")

    }

    private fun drawMask(canvas: Canvas, mask: Bitmap){

        val maskRect = Rect(
                0,
                0,
                mask.width,
                mask.height)

        Log.d("drawing Mask:", mask.getPixel(60,50).toString() );
        canvas.drawBitmap(mask, null, maskRect, null)
    }

}