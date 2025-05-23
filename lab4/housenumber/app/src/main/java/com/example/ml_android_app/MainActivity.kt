package com.example.ml_android_app

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import com.example.ml_android_app.image_processing_model.ImageModel
import java.io.File
import java.io.FileNotFoundException
import java.io.IOException

class MainActivity : ComponentActivity() {
    private lateinit var imageModel: ImageModel

    private lateinit var photoText: TextView
    private lateinit var photoImageView: ImageView

    private lateinit var loadPhotoLauncher: ActivityResultLauncher<Intent>
    private lateinit var takePhotoLauncher: ActivityResultLauncher<Intent>

    private lateinit var thresholdSeekBar: SeekBar
    private lateinit var thresholdTextView: TextView
    private var currentThreshold: Float = 0.5f


    private lateinit var takenPhotoUri: Uri

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        imageModel = ImageModel(assets)

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

        photoImageView = findViewById(R.id.ivPhoto)
        photoText = findViewById(R.id.tvNumber)
        findViewById<Button>(R.id.btTake).setOnClickListener { takePhoto() }
        findViewById<Button>(R.id.btLoad).setOnClickListener { loadPhoto() }
        findViewById<Button>(R.id.btClear).setOnClickListener { clear() }

        thresholdSeekBar = findViewById(R.id.thresholdSeekBar)
        thresholdTextView = findViewById(R.id.tvThresholdLabel)

        thresholdSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                currentThreshold = progress / 100f
                thresholdTextView.text = "Порог чувствительности: %.2f".format(currentThreshold).replace(',','.')
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })


    }

    private fun clear() {
        photoText.text = "-"
        photoImageView.setImageResource(R.drawable.ic_placeholder)
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
        if (uri == null) return

        try {
            contentResolver.openInputStream(uri)?.use { inputStream ->
                val imageBitmap = BitmapFactory.decodeStream(inputStream)

                contentResolver.openInputStream(uri)?.use { exifInputStream ->
                    val exif = ExifInterface(exifInputStream)
                    val rotatedBitmap = rotateBitmap(imageBitmap, exif)
                    photoImageView.setImageBitmap(rotatedBitmap)
                    val predictedNumber = predictNumber(rotatedBitmap)
                    photoText.text = if (predictedNumber.isNotEmpty()) predictedNumber else "-"
                }
            }
        } catch (e: FileNotFoundException) {
            e.printStackTrace()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    private fun predictNumber(bitmap: Bitmap): String {
        val digitRegions = imageModel.getDigitRegions(bitmap, currentThreshold)
        val digits = mutableListOf<Int>()

        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint().apply {
            color = Color.parseColor("#008b8b")
            style = Paint.Style.STROKE
            strokeWidth = 20f
        }

        for (region in digitRegions) {
            val predictedDigit = imageModel.predictDigit(region.bitmap)
            digits.add(predictedDigit)
            canvas.drawRect(region.boundingBox, paint)
        }

        photoImageView.setImageBitmap(mutableBitmap)

        return digits.joinToString("") { it.toString() }
    }

    private fun rotateBitmap(bitmap: Bitmap, exif: ExifInterface): Bitmap {
        val orientation = exif.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        )

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
