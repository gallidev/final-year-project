package com.example.videosegmentation

import android.content.Context
import android.graphics.*
import com.otaliastudios.cameraview.CameraView
import android.os.SystemClock
import android.renderscript.*
import android.renderscript.Allocation
import com.dailystudio.app.utils.BitmapUtils


class ImageProcessor(
        private val cameraView: CameraView,
        private val overlayViewMask: OverlayView,
        private val activity: SegmentationActivity,
        private val contextApplication: Context) {

    private val imageYUVsize = 2880000
    private val frameWIDTH = 1200
    private val frameHEIGHT = 1600

    private var lastFrame: Bitmap? = null


    fun startProcessing() {

        val rs = RenderScript.create(contextApplication)
        val yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))

        val yuvType = Type.Builder(rs, Element.U8(rs)).setX(imageYUVsize)
        val input = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT)

        val rgbaType = Type.Builder(rs, Element.RGBA_8888(rs)).setX(frameWIDTH).setY(frameHEIGHT)
        val out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT)


        // Getting frames from camera view
        cameraView.addFrameProcessor { frame ->

            if (frame.size != null ) {
                //Log.d("Frame", "start processing frame")
                val startTime = SystemClock.uptimeMillis()

                val rotatedYuv = rotateYUV420Degree270(frame.data, frame.size.width, frame.size.height)

                val bitmap = Bitmap.createBitmap(frame.size.height, frame.size.width, Bitmap.Config.ARGB_8888)


                input.copyFrom(rotatedYuv)

                yuvToRgbIntrinsic.setInput(input)
                yuvToRgbIntrinsic.forEach(out)

                //val bmData = renderScriptNV21ToRGBA888(rs,
                //        contextApplication,
                //        frame.size.height,
                //        frame.size.width,
                //        rotatedYuv)

                out.copyTo(bitmap)
                //it is essential to destroy the allocation object to prevent memory leaks

                setLastFrame(bitmap)

                val endTime = SystemClock.uptimeMillis()
                //Log.d("TIME", "conversion from YUV bitmap: " + java.lang.Long.toString(endTime - startTime))

                //Log.d("rotation in float", rotation.toFloat().toString());
                //val startTimeBitmap = SystemClock.uptimeMillis()
                //val rotatedBitmap = rotateFlipImage(bitmap, 270.0f)
                //val rotatedBitmap = bitmap
//                overlayView.mask = rotatedBitmap
//                overlayView.invalidate()

                val w = bitmap.width
                val h = bitmap.height
                // Log.d("decoded frame dimen:", w.toString() + " - " + h.toString())


                val resizeRatio = UnetPortraits.getInputSize() / Math.max(bitmap.width, bitmap.height)
                val rw = Math.round(w * resizeRatio)
                val rh = Math.round(h * resizeRatio)
                //Log.d("Resize bitmap", "ratio: " + resizeRatio.toString() + " -> " + rw + " - " + rh)
//                Log.debug("resize bitmap: ratio = %f, [%d x %d] -> [%d x %d]",
//                        resizeRatio, w, h, rw, rh)

                val resized = ImageUtils.tfResizeBilinear(bitmap, rw, rh, 270.0f)

                val endTimeBitmap = SystemClock.uptimeMillis()
                //Log.d("Frame", "Completed frame prep with size" + resized.width + "- " + resized.height)

                val startTimeInference = SystemClock.uptimeMillis()

                //Log.d("TIME","inference start")
                ModelManager.setIsProcessing(true)
                var mask = ModelManager.getInstance().segment(resized)
                //Log.d("TIME","inference end")
                ModelManager.setIsProcessing(false)
                activity.onImageSegmentationEnd()
                val endTimeInference = SystemClock.uptimeMillis()

                //Log.d("TIME", "Inference Completed in " + java.lang.Long.toString(endTimeInference - startTimeInference))

                if(mask != null){

                    //resize the mask to 3x4 aspect ratio if required
                    mask = BitmapUtils.createClippedBitmap(mask,
                            (mask.width - rw) / 2,
                            (mask.height - rh) / 2,
                            rw, rh)

                    overlayViewMask.mask = mask
                    overlayViewMask.invalidate()

                    //Log.d("Mask", "sent Mask")
                    activity.showPerformance(ModelManager.getModelName(), java.lang.Long.toString(endTime - startTime),
                            java.lang.Long.toString(endTimeInference - startTimeInference))

                }
            }
        }
    }

    //Converts YUV image into RGB in ~10ms
    private fun renderScriptNV21ToRGBA888(rs:RenderScript, context: Context, width: Int, height: Int, nv21: ByteArray): Allocation {
        //val rs = RenderScript.create(context)
        val yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))

        val yuvType = Type.Builder(rs, Element.U8(rs)).setX(nv21.size)
        val input = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT)

        val rgbaType = Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height)
        val out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT)

        input.copyFrom(nv21)

        yuvToRgbIntrinsic.setInput(input)
        yuvToRgbIntrinsic.forEach(out)

        return out
    }

    private fun rotateYUV420Degree180(data: ByteArray, imageWidth: Int, imageHeight: Int): ByteArray {
        val yuv = ByteArray(imageWidth * imageHeight * 3 / 2)
        var i = 0
        var count = 0
        i = imageWidth * imageHeight - 1
        while (i >= 0) {
            yuv[count] = data[i]
            count++
            i--
        }
        i = imageWidth * imageHeight * 3 / 2 - 1
        i = imageWidth * imageHeight * 3 / 2 - 1
        while (i >= imageWidth * imageHeight) {
            yuv[count++] = data[i - 1]
            yuv[count++] = data[i]
            i -= 2
        }
        return yuv
    }

    private fun rotateYUV420Degree270(data: ByteArray, imageWidth: Int,
                              imageHeight: Int): ByteArray {
        val yuv = ByteArray(imageWidth * imageHeight * 3 / 2)
        var nWidth = 0
        var nHeight = 0
        var wh = 0
        var uvHeight = 0
        if (imageWidth != nWidth || imageHeight != nHeight) {
            nWidth = imageWidth
            nHeight = imageHeight
            wh = imageWidth * imageHeight
            uvHeight = imageHeight shr 1// uvHeight = height / 2
        }
        // ??Y
        var k = 0
        for (i in 0 until imageWidth) {
            var nPos = 0
            for (j in 0 until imageHeight) {
                yuv[k] = data[nPos + i]
                k++
                nPos += imageWidth
            }
        }
        var i = 0
        while (i < imageWidth) {
            var nPos = wh
            for (j in 0 until uvHeight) {
                yuv[k] = data[nPos + i]
                yuv[k + 1] = data[nPos + i + 1]
                k += 2
                nPos += imageWidth
            }
            i += 2
        }
        return rotateYUV420Degree180(yuv, imageWidth, imageHeight)
    }


    @Synchronized fun getLastFrame():Bitmap? {
        return lastFrame
    }

    @Synchronized private fun setLastFrame(frame: Bitmap){
        lastFrame = frame
    }

}