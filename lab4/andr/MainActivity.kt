package com.example.ml_android_app

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import com.example.ml_android_app.image_processing_model.ImageDetectionModel
import com.example.ml_android_app.image_processing_model.ImagePredictionModel
import java.io.File
import java.io.FileNotFoundException
import java.io.IOException


class MainActivity : ComponentActivity() {
    private lateinit var imagePredictionModel: ImagePredictionModel
    private lateinit var imageDetectionModel: ImageDetectionModel

    private lateinit var photoText: TextView
    private lateinit var photoImageView: ImageView

    private lateinit var loadPhotoLauncher: ActivityResultLauncher<Intent>
    private lateinit var takePhotoLauncher: ActivityResultLauncher<Intent>

    private lateinit var takenPhotoUri: Uri

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        imagePredictionModel = ImagePredictionModel(assets)
        imageDetectionModel = ImageDetectionModel(assets)

        setContentView(R.layout.activity_main)

        takePhotoLauncher =
            registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
                if (result.resultCode == RESULT_OK) {
                    processImageUri(takenPhotoUri)
                }
            }

        loadPhotoLauncher =
            registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
                if (result.resultCode == RESULT_OK) {
                    processImageUri(result.data?.data)
                }
            }

        photoImageView = findViewById(R.id.photoImageView)
        photoText = findViewById(R.id.photoText)
        findViewById<Button>(R.id.takePhotoButton).setOnClickListener { takePhoto() }
        findViewById<Button>(R.id.loadPhotoButton).setOnClickListener { loadPhoto() }
    }

    private fun takePhoto() {
        val tempFile = File.createTempFile("temp_image_", ".jpg", cacheDir)
        takenPhotoUri = FileProvider.getUriForFile(this, "${packageName}.fileprovider", tempFile)

        val takePhotoIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        takePhotoIntent.putExtra(MediaStore.EXTRA_OUTPUT, takenPhotoUri)

        if (takePhotoIntent.resolveActivity(packageManager) != null) {
            takePhotoLauncher.launch(takePhotoIntent)
        }
    }

    private fun loadPhoto() {
        val loadPhotoIntent =
            Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        loadPhotoLauncher.launch(loadPhotoIntent)
    }

    private fun processImageUri(uri: Uri?) {
        if (uri == null) {
            return
        }
        try {
            contentResolver.openInputStream(uri)?.use { inputStream ->
                val imageBitmap = BitmapFactory.decodeStream(inputStream)

                contentResolver.openInputStream(uri)?.use { exifInputStream ->
                    val exif = ExifInterface(exifInputStream)
                    val rotatedBitmap = rotateBitmap(imageBitmap, exif)
                    photoImageView.setImageBitmap(rotatedBitmap)
                    val predictedNumber = predictNumber(rotatedBitmap)
                    photoText.text = getString(R.string.house_number, predictedNumber)
                }
            }
        } catch (e: FileNotFoundException) {
            e.printStackTrace()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    private fun predictNumber(bitmap: Bitmap): String {
        val digitBitmaps = imageDetectionModel.getDigitBitmaps(bitmap)
        val digits = mutableListOf<Int>()

        for (digitBitmap in digitBitmaps) {
            val predictedDigit = imagePredictionModel.predictDigit(digitBitmap)
            digits.add(predictedDigit)
        }

        return digits.joinToString("") { it.toString() }
    }

    private fun rotateBitmap(bitmap: Bitmap, exif: ExifInterface): Bitmap {
        val orientation =
            exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)

        val rotationDegrees = when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }

        return if (rotationDegrees != 0f) {
            val matrix = Matrix().apply { postRotate(rotationDegrees) }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }
    }
}