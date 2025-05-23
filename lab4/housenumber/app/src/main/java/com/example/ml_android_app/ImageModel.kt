package com.example.ml_android_app.image_processing_model

import android.content.res.AssetFileDescriptor
import android.content.res.AssetManager
import android.graphics.*
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ImageModel(private val assets: AssetManager) {

    private val detectionModel = loadModelFile("yolov5.tflite")
    private val predictionModel = loadModelFile("housenumber_model_32.tflite")

    private val detectionInterpreter = Interpreter(detectionModel)
    private val predictionInterpreter = Interpreter(predictionModel)

    private fun loadModelFile(name: String): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = assets.openFd(name)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    private fun preprocessImage(bitmap: Bitmap, height: Int, width: Int, colorNumber: Int): Array<Array<Array<FloatArray>>> {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val input = Array(1) { Array(height) { Array(width) { FloatArray(colorNumber) } } }

        for (i in 0 until height) {
            for (j in 0 until width) {
                val pixel = resizedBitmap.getPixel(j, i)
                val r = (pixel shr 16 and 0xFF) / 255.0f
                val g = (pixel shr 8 and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f

                if (colorNumber == 1) {
                    input[0][i][j][0] = (0.299 * r + 0.587 * g + 0.114 * b).toFloat()
                } else {
                    input[0][i][j][0] = r
                    input[0][i][j][1] = g
                    input[0][i][j][2] = b
                }
            }
        }

        return input
    }

    private data class BoundingBox(
        val centerX: Float,
        val centerY: Float,
        val width: Float,
        val height: Float,
        val confidence: Float
    )

    data class DigitRegion(
        val bitmap: Bitmap,
        val boundingBox: Rect
    )

    fun getDigitRegions(bitmap: Bitmap, percent: Float): List<DigitRegion> {
        val input = preprocessImage(bitmap, 640, 640, 3)
        val output = Array(1) { Array(25200) { FloatArray(15) } }

        detectionInterpreter.run(input, output)

        val boundingBoxes = mutableListOf<BoundingBox>()
        for (i in output[0].indices) {
            val confidence = output[0][i][4]
            if (confidence > percent) {
                val x = output[0][i][0]
                val y = output[0][i][1]
                val width = output[0][i][2]
                val height = output[0][i][3]
                boundingBoxes.add(BoundingBox(x, y, width, height, confidence))
            }
        }

        return nonMaximumSuppression(boundingBoxes)
            .sortedBy { it.centerX }
            .mapNotNull { box ->
                val centerX = box.centerX * bitmap.width
                val centerY = box.centerY * bitmap.height
                val boxWidth = box.width * bitmap.width
                val boxHeight = box.height * bitmap.height

                val left = (centerX - boxWidth / 2).toInt().coerceIn(0, bitmap.width)
                val top = (centerY - boxHeight / 2).toInt().coerceIn(0, bitmap.height)
                val right = (centerX + boxWidth / 2).toInt().coerceIn(0, bitmap.width)
                val bottom = (centerY + boxHeight / 2).toInt().coerceIn(0, bitmap.height)

                val rect = Rect(left, top, right, bottom)
                try {
                    val digitBitmap = Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.width(), rect.height())
                    DigitRegion(digitBitmap, rect)
                } catch (e: IllegalArgumentException) {
                    null
                }
            }
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

    fun predictDigit(bitmap: Bitmap): Int {
        val input = preprocessImage(bitmap, 32, 32, 1)
        val output = Array(1) { FloatArray(10) }

        predictionInterpreter.run(input, output)
        return output[0].indices.maxByOrNull { output[0][it] } ?: -1
    }
}
