package com.example.ml_android_app.image_processing_model

import android.content.res.AssetFileDescriptor
import android.content.res.AssetManager
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel


open class ImageProcessingModel(
    assets: AssetManager,
    modelFileName: String,
    private val height: Int,
    private val width: Int,
    private val colorNumber: Int
) {

    private var model: MappedByteBuffer

    init {
        model = loadModelFile(assets, modelFileName)
    }

    private fun loadModelFile(assets: AssetManager, name: String): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = assets.openFd(name)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    protected fun getInterpreter(): Interpreter {
        return Interpreter(model)
    }

    protected fun preprocessImage(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
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
}