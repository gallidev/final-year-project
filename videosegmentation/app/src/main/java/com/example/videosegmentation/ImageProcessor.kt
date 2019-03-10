package com.example.videosegmentation

import android.content.Context
import android.graphics.*
import com.otaliastudios.cameraview.CameraView
import android.util.Log
import android.os.SystemClock
import android.renderscript.*
import android.renderscript.Allocation


class ImageProcessor(private val cameraView: CameraView,
                     private val overlayViewMask: OverlayView,
                     private val activity: SegmentationActivity,
                     private val contextApplication: Context) {

    var backgroundImage: Bitmap = BitmapFactory.decodeResource(contextApplication.resources, R.drawable.beach)

    fun startProcessing() {

        // Getting frames from camera view
        cameraView.addFrameProcessor { frame ->

            if (frame.size != null) {

                Log.d("Frame", "start processing frame")
                val rotation = frame.rotation / 90
                if (rotation / 2 == 0) {
                    overlayViewMask.previewWidth = cameraView.previewSize?.width
                    overlayViewMask.previewHeight = cameraView.previewSize?.height
                } else {
                    overlayViewMask.previewWidth = cameraView.previewSize?.height
                    overlayViewMask.previewHeight = cameraView.previewSize?.width
                }
                // to convert the image I found this online, gosh https://github.com/natario1/CameraView/issues/310
                // this might slow down things, I should time it... gosh...

                val startTime = SystemClock.uptimeMillis()

                val rotatedYuv = rotateYUV420Degree270(frame.data, frame.size.width, frame.size.height)
                val bitmap = Bitmap.createBitmap(frame.size.height, frame.size.width, Bitmap.Config.ARGB_8888)
                val bmData = renderScriptNV21ToRGBA888(
                        contextApplication,
                        frame.size.height,
                        frame.size.width,
                        rotatedYuv)
                bmData.copyTo(bitmap)

                val endTime = SystemClock.uptimeMillis()
                Log.d("TIME", "conversion from YUV bitmap: " + java.lang.Long.toString(endTime - startTime))

                //Log.d("rotation in float", rotation.toFloat().toString());
                val startTimeBitmap = SystemClock.uptimeMillis()
                //val rotatedBitmap = rotateFlipImage(bitmap, 270.0f)
                val rotatedBitmap = bitmap
//                overlayView.mask = rotatedBitmap
//                overlayView.invalidate()

                val w = rotatedBitmap.width
                val h = rotatedBitmap.height
               // Log.d("decoded frame dimen:", w.toString() + " - " + h.toString())


                val resizeRatio = UnetPortraits.getInputSize() / Math.max(rotatedBitmap.width, rotatedBitmap.height)
                val rw = Math.round(w * resizeRatio)
                val rh = Math.round(h * resizeRatio)
                //Log.d("Resize bitmap", "ratio: " + resizeRatio.toString() + " -> " + rw + " - " + rh)
//                Log.debug("resize bitmap: ratio = %f, [%d x %d] -> [%d x %d]",
//                        resizeRatio, w, h, rw, rh)

                val resized = ImageUtils.tfResizeBilinear(rotatedBitmap, rw, rh, 270.0f)

                val endTimeBitmap = SystemClock.uptimeMillis()
                //Log.d("Frame", "Completed frame prep with size" + resized.width + "- " + resized.height)

                val startTimeInference = SystemClock.uptimeMillis()

                var mask = SegmentationModel.getInstance().segment(resized)

                val endTimeInference = SystemClock.uptimeMillis()

                Log.d("TIME", "Inference Completed in " + java.lang.Long.toString(endTimeInference - startTimeInference))

                if(mask != null){

                    //mask = cropBitmapWithMask(backgroundImage, mask)
                    overlayViewMask.mask = mask
                    overlayViewMask.invalidate()

                    Log.d("Mask", "sent Mask")
                    activity.showPerformance(java.lang.Long.toString(endTime - startTime),
                            java.lang.Long.toString(endTimeBitmap - startTimeBitmap),
                            java.lang.Long.toString(endTimeInference - startTimeInference))

                }
            }
        }
    }

    //Converts YUV image into RGB in ~10ms
    private fun renderScriptNV21ToRGBA888(context: Context, width: Int, height: Int, nv21: ByteArray): Allocation {
        val rs = RenderScript.create(context)
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


    fun cropBitmapWithMask(original: Bitmap, mask: Bitmap): Bitmap? {
        if (original == null
                || mask == null) {
            return null;
        }

        var w = original.getWidth()
        var h = original.getHeight()
        if (w <= 0 || h <= 0) {
            return null;
        }

        var cropped = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)


        var canvas = Canvas(cropped)
        var paint = Paint(Paint.ANTI_ALIAS_FLAG)


        paint.setXfermode(PorterDuffXfermode(PorterDuff.Mode.DST_IN))

        canvas.drawBitmap(original, 0.0f, 0.0f, null)
        canvas.drawBitmap(mask, 0.0f, 0.0f, paint)
        paint.setXfermode(null)

        return cropped;
    }

}