package com.example.videosegmentation

import android.graphics.*
import com.otaliastudios.cameraview.CameraView
import android.util.Log
import java.io.ByteArrayOutputStream
import android.graphics.Bitmap
import android.os.SystemClock
import com.dailystudio.app.utils.BitmapUtils


/**
 *
 * FaceProcessor takes the camera frames from CameraView and uses FirebaseVisionFaceDetector
 * to detect the face, and then pass the detected face info to OverlayView so it can draw bitmaps on the face
 *
 * Created by Qichuan on 21/6/18.
 */
class ImageProcessor(private val cameraView: CameraView,
                     private val overlayViewMask: OverlayView,
                     private val overlayViewCropped: OverlayView) {

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

                val out = ByteArrayOutputStream()
                val yuvImage = YuvImage(frame.data, ImageFormat.NV21, frame.size.width, frame.size.height, null)
                yuvImage.compressToJpeg(Rect(0, 0, frame.size.width, frame.size.height), 90, out)
                val imageBytes = out.toByteArray()
                val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

                val endTime = SystemClock.uptimeMillis()
                Log.d("TIME", "conversion from YUV bitmap: " + java.lang.Long.toString(endTime - startTime))

                //Log.d("rotation in float", rotation.toFloat().toString());
                val startTimeBitmap = SystemClock.uptimeMillis()
                val rotatedBitmap = rotateFlipImage(bitmap, 270.0f)
//                overlayView.mask = rotatedBitmap
//                overlayView.invalidate()

                val w = rotatedBitmap.width
                val h = rotatedBitmap.height
               // Log.d("decoded frame dimen:", w.toString() + " - " + h.toString())

                //val resizeRatio = DeeplabGPU.getInputSize() / Math.max(rotatedBitmap.width, rotatedBitmap.height)
                val resizeRatio = UnetCPU.getInputSize() / Math.max(rotatedBitmap.width, rotatedBitmap.height)
                val rw = Math.round(w * resizeRatio)
                val rh = Math.round(h * resizeRatio)
                //Log.d("Resize bitmap", "ratio: " + resizeRatio.toString() + " -> " + rw + " - " + rh)
//                Log.debug("resize bitmap: ratio = %f, [%d x %d] -> [%d x %d]",
//                        resizeRatio, w, h, rw, rh)

                val resized = ImageUtils.tfResizeBilinear(rotatedBitmap, rw, rh)

                val endTimeBitmap = SystemClock.uptimeMillis()

                Log.d("TIME", "Bitmap for Model " + java.lang.Long.toString(endTimeBitmap - startTimeBitmap))

                //Log.d("Frame", "Completed frame prep with size" + resized.width + "- " + resized.height)

                val startTimeInference = SystemClock.uptimeMillis()

                // var mask = SegmentationModel.getInstance().segment(resized)
                var mask = SegmentationModel.getUnetInstance().segment(resized)

                val endTimeInference = SystemClock.uptimeMillis()

                Log.d("TIME", "Inference Completed in " + java.lang.Long.toString(endTimeInference - startTimeInference))


                mask = BitmapUtils.createClippedBitmap(mask,
                        (mask.width - rw) / 2,
                        (mask.height - rh) / 2,
                        rw, rh)

                overlayViewMask.mask = mask
                overlayViewMask.invalidate()

                Log.d("Mask", "sent Mask")

                mask = BitmapUtils.scaleBitmap(mask, w, h)

                val cropped = cropBitmapWithMask(rotatedBitmap, mask)
                Log.d("cropped", "completed Cropped")
                overlayViewCropped.mask = cropped
                overlayViewCropped.invalidate()
                Log.d("Cropped", "sent Cropped")
            }
        }
    }

    fun rotateFlipImage(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix()
        matrix.postScale(1f, -1f, source.width/2f,source.height/2f)
        matrix.postRotate(angle)
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height,
                matrix, true)
    }

    private fun cropBitmapWithMask(original: Bitmap?, mask: Bitmap?): Bitmap? {
        if (original == null || mask == null) {
            return null
        }

        val w = original.width
        val h = original.height
        if (w <= 0 || h <= 0) {
            return null
        }

        val cropped = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)


        val canvas = Canvas(cropped)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)

        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN)
        canvas.drawBitmap(original, 0f, 0f, null)
        canvas.drawBitmap(mask, 0f, 0f, paint)
        paint.xfermode = null

        return cropped
    }


}