package com.abhishekgarala

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.abhishekgarala.databinding.ActivityMainBinding
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.selectImageButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK)
            intent.type = "image/*"
            startActivityForResult(intent, 100)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 100 && resultCode == Activity.RESULT_OK) {
            val imageUri: Uri? = data?.data
            imageUri?.let {
                val inputStream: InputStream? = contentResolver.openInputStream(it)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                binding.imageView.setImageBitmap(bitmap)
                performObjectDetection(bitmap)
            }
        }
    }

    private fun performObjectDetection(bitmap: Bitmap) {
        val model = ObjectDetector.ObjectDetectorOptions.builder()
            .setMaxResults(10)
            .setScoreThreshold(0.5f)
            .build()
        val objectDetector = ObjectDetector.createFromFileAndOptions(
            this,
            "mobilenetv1.tflite",
            model
        )

        val inputImage = TensorImage.fromBitmap(bitmap)
        val results = objectDetector.detect(inputImage)

        val resultBitmap = drawBoundingBoxes(bitmap, results)
        binding.resultImageView.setImageBitmap(resultBitmap)

        val resultText = results.joinToString("\n") {
            "Object: ${it.categories.first().label}, Confidence: ${(it.categories.first().score * 100).toInt()}%"
        }
        binding.resultTextView.text = resultText
    }

    private fun drawBoundingBoxes(bitmap: Bitmap, detections: List<Detection>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val boxPaint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 6f
        }
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 32f
            style = Paint.Style.FILL
        }
        val textBackgroundPaint = Paint().apply {
            color = Color.BLACK
            style = Paint.Style.FILL
        }

        detections.forEach { detection ->
            val box = detection.boundingBox
            val label = detection.categories.first().label
            val confidence = (detection.categories.first().score * 100).toInt()

            boxPaint.color = Color.RED
            canvas.drawRect(box, boxPaint)

            val text = "$label ($confidence%)"
            val textWidth = textPaint.measureText(text)
            val textHeight = textPaint.textSize
            canvas.drawRect(
                box.left,
                box.top - textHeight,
                box.left + textWidth,
                box.top,
                textBackgroundPaint
            )
            canvas.drawText(text, box.left, box.top, textPaint)
        }

        return mutableBitmap
    }

}