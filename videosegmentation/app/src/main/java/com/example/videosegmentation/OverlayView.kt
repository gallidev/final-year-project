package com.example.videosegmentation

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.util.Log
import android.view.View
import android.graphics.BitmapFactory
import android.graphics.Bitmap



/**
 * A overlay view that draws thug life glasses and cigarette bitmaps on top of a detected face
 *
 * Created by Qichuan on 21/6/18.
 */
class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {


    var mask: Bitmap? = null
    var oldMask: Bitmap? = null
    var backgroundImage: Bitmap = BitmapFactory.decodeResource(resources, R.drawable.beach)



    override fun onLayout(changed: Boolean, left: Int, top: Int, right: Int, bottom: Int) {
        super.onLayout(changed, left, top, right, bottom)
        // you need to disable hardware acceleration for this
        //https@ //stackoverflow.com/questions/18387814/drawing-on-canvas-porterduff-mode-clear-draws-black-why
        setLayerType(View.LAYER_TYPE_SOFTWARE, null)
    }

    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)

        // Create local variables here so they cannot not be changed anywhere else
        val maskFixed = mask

        //Log.d("draw", "checking if the mask or old mask are not null")
        if(maskFixed != null && canvas != null){

            val maskRect = Rect(
                    0,
                    0,
                    canvas.width,
                    canvas.height)


            var paint = Paint()
            paint.setAntiAlias(true)
            paint.setFilterBitmap(true)
            paint.setDither(true)

            paint.setXfermode(PorterDuffXfermode(PorterDuff.Mode.DST_IN))

            //Log.d("drawing Mask:", maskFixed.getPixel(60,50).toString());
            //canvas.drawBitmap(maskFixed, null, maskRect, Paint(Paint.FILTER_BITMAP_FLAG))

            canvas.drawBitmap(backgroundImage, null, maskRect, null)
            canvas.drawBitmap(mask, null, maskRect, paint)
            paint.setXfermode(null)


        }
    }

}