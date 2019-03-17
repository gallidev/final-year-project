package com.example.videosegmentation

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import android.graphics.BitmapFactory
import android.graphics.Bitmap
import android.graphics.BlurMaskFilter.Blur
import android.graphics.BlurMaskFilter
import android.graphics.PorterDuff


class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {


    var mask: Bitmap? = null
    var backgroundImages: Array<Bitmap> = arrayOf(BitmapFactory.decodeResource(resources, R.drawable.beach) ,
                BitmapFactory.decodeResource(resources, R.drawable.tour_eiffel),
                BitmapFactory.decodeResource(resources, R.drawable.tajmahal),
                BitmapFactory.decodeResource(resources, R.drawable.green_matrix_min)
    )
    var indexImage = 0

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


            val maskWithBlur = highlightImage(maskFixed)

            val maskRect = Rect(
                    0,
                    0,
                    canvas.width,
                    canvas.height)


            var paint = Paint()
            paint.setAntiAlias(true)
            paint.setFilterBitmap(true)
            paint.setDither(true)
            //

            paint.setXfermode(PorterDuffXfermode(PorterDuff.Mode.DST_IN))
            //Log.d("drawing Mask:", maskFixed.getPixel(60,50).toString());
            //canvas.drawBitmap(maskFixed, null, maskRect, Paint(Paint.FILTER_BITMAP_FLAG))

            if(indexImage == backgroundImages.size){
                canvas.drawColor(Color.WHITE)
            }else{
                canvas.drawBitmap(backgroundImages[indexImage], null, maskRect, null)
            }

            canvas.drawBitmap(maskWithBlur, null, maskRect, paint)
            paint.setXfermode(null)


        }
    }

    fun nextBackground(){
        indexImage ++
        if(indexImage == backgroundImages.size + 1){
            indexImage = 0
        }
    }


    private fun highlightImage(src: Bitmap): Bitmap {
        // create new bitmap, which will be painted and becomes result image
        val bmOut = Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888)
        // setup canvas for painting
        val canvas = Canvas(bmOut)
        // setup default color
        canvas.drawColor(Color.WHITE, PorterDuff.Mode.CLEAR)
        // create a blur paint for capturing alpha
        val ptBlur = Paint()
        ptBlur.maskFilter = BlurMaskFilter(1f, Blur.NORMAL)
        val offsetXY = IntArray(2)
        // capture alpha into a bitmap
        val bmAlpha = src.extractAlpha(ptBlur, offsetXY)
        // create a color paint
        val ptAlphaColor = Paint()
        ptAlphaColor.color = -0x1
        // paint color for captured alpha region (bitmap)
        canvas.drawBitmap(bmAlpha, offsetXY[0].toFloat(), offsetXY[1].toFloat(), ptAlphaColor)
        //2 times makes the blur stronger
        canvas.drawBitmap(bmAlpha, offsetXY[0].toFloat(), offsetXY[1].toFloat(), ptAlphaColor)
        // free memory
        bmAlpha.recycle()

        // paint the image source
        canvas.drawBitmap(src, 0f, 0f, null)

        // return out final image
        return bmOut
    }

    fun saveScreen(frame: Bitmap): Bitmap {
        // Create local variables here so they cannot not be changed anywhere else
        val maskFixed = mask

        val processedMask = Bitmap.createBitmap(frame.width, frame.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(processedMask)


        //Log.d("draw", "checking if the mask or old mask are not null")
        if(maskFixed != null ){
            val maskWithBlur = highlightImage(maskFixed)

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

            canvas.drawBitmap(backgroundImages[indexImage], null, maskRect, null)
            canvas.drawBitmap(maskWithBlur, null, maskRect, paint)
            paint.setXfermode(null)

            // at this point processedMask should be the size of the screen with the image as background
            val canvasFinal = Canvas(frame)
            canvasFinal.drawBitmap(processedMask, null, maskRect, null)

            return frame
        }
        
        return frame
    }

}