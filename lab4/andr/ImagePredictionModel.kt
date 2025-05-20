package com.example.ml_android_app.image_processing_model

import android.content.res.AssetManager
import android.graphics.Bitmap


class ImagePredictionModel(assets: AssetManager): ImageProcessingModel(
    assets,
    modelFileName = "svhn_model.tflite",
    height = 32,
    width = 32,
    colorNumber = 1
) {
    fun predictDigit(bitmap: Bitmap): Int {
        val input = preprocessImage(bitmap)
        val output = Array(1) { FloatArray(10) }

        getInterpreter().run(input, output)
        return output[0].indices.maxByOrNull { output[0][it] } ?: -1
    }
}