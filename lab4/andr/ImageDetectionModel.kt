package com.example.ml_android_app.image_processing_model

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect


class ImageDetectionModel(assets: AssetManager) : ImageProcessingModel(
    assets,
    modelFileName = "yolov5_best-fp16.tflite",
    height = 640,
    width = 640,
    colorNumber = 3
) {
    private data class BoundingBox(
        val centerX: Float,
        val centerY: Float,
        val width: Float,
        val height: Float,
        val confidence: Float
    )

    fun getDigitBitmaps(bitmap: Bitmap): List<Bitmap> {
        val input = preprocessImage(bitmap)
        val output = Array(1) { Array(25200) { FloatArray(15) } }

        getInterpreter().run(input, output)
        val boundingBoxes = mutableListOf<BoundingBox>()

        for (i in output[0].indices) {
            val confidence = output[0][i][4]
            if (confidence > 0.5) {
                val x = output[0][i][0]
                val y = output[0][i][1]
                val width = output[0][i][2]
                val height = output[0][i][3]
                boundingBoxes.add(BoundingBox(x, y, width, height, confidence))
            }
        }

        return nonMaximumSuppression(boundingBoxes)
            .sortedBy { it.centerX }
            .map { cropBitmap(bitmap, it) }
    }

    private fun nonMaximumSuppression(boxes: List<BoundingBox>): List<BoundingBox> {
        var sortedBoxes = boxes.sortedByDescending { it.confidence }
        val selectedBoxes = mutableListOf<BoundingBox>()

        while (sortedBoxes.isNotEmpty()) {
            val currentBox = sortedBoxes.first()
            selectedBoxes.add(currentBox)

            sortedBoxes = sortedBoxes.drop(1).filter { box ->
                calculateIntersectionOverUnion(currentBox, box) < 0.5
            }
        }

        return selectedBoxes
    }

    private fun calculateIntersectionOverUnion(boxA: BoundingBox, boxB: BoundingBox): Float {
        val xA = maxOf(boxA.centerX, boxB.centerX)
        val yA = maxOf(boxA.centerY, boxB.centerY)
        val xB = minOf(boxA.centerX + boxA.width, boxB.centerX + boxB.width)
        val yB = minOf(boxA.centerY + boxA.height, boxB.centerY + boxB.height)

        val interArea = maxOf(0f, xB - xA) * maxOf(0f, yB - yA)
        val boxAArea = boxA.width * boxA.height
        val boxBArea = boxB.width * boxB.height

        return interArea / (boxAArea + boxBArea - interArea)
    }

    private fun cropBitmap(bitmap: Bitmap, box: BoundingBox): Bitmap {
        val size = maxOf(bitmap.width * box.width, bitmap.height * box.height).toInt()
        val halfSize = size / 2
        val cropX = (box.centerX * bitmap.width).toInt() - halfSize
        val cropY = (box.centerY * bitmap.height).toInt() - halfSize

        val croppedBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(croppedBitmap)
        canvas.drawColor(Color.BLACK)

        val srcRect = Rect(
            maxOf(cropX, 0),
            maxOf(cropY, 0),
            minOf(cropX + size, bitmap.width),
            minOf(cropY + size, bitmap.height)
        )

        val destRect = Rect(0, 0, size, size)
        canvas.drawBitmap(bitmap, srcRect, destRect, null)
        return croppedBitmap
    }
}