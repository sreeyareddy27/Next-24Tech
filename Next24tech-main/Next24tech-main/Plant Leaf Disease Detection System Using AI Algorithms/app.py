from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
from PIL import Image
import numpy as np
import sys
import tensorflow as tf
import os

# Load the trained model
model = tf.keras.models.load_model('leaf_disease_model.h5')

# Define class names and disease descriptions
class_names = {
    'Pepper__bell___Bacterial_spot': 'Pepper - Bacterial Spot',
    'Pepper__bell___healthy': 'Pepper - Healthy',
    'Potato___Early_blight': 'Potato - Early Blight',
    'Potato___Late_blight': 'Potato - Late Blight',
    'Potato___healthy': 'Potato - Healthy',
    'Tomato_Bacterial_spot': 'Tomato - Bacterial Spot',
    'Tomato_Early_blight': 'Tomato - Early Blight',
    'Tomato_Late_blight': 'Tomato - Late Blight',
    'Tomato_Leaf_Mold': 'Tomato - Leaf Mold',
    'Tomato_Septoria_leaf_spot': 'Tomato - Septoria Leaf Spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato - Spider Mites Two-Spotted Spider Mite',
    'Tomato__Target_Spot': 'Tomato - Target Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Tomato - Yellow Leaf Curl Virus',
    'Tomato__Tomato_mosaic_virus': 'Tomato - Mosaic Virus',
    'Tomato_healthy': 'Tomato - Healthy'
}

disease_descriptions = {
    'Pepper - Bacterial Spot': "Caused by the bacterium Xanthomonas campestris pv. vesicatoria, it leads to dark, water-soaked lesions on leaves and fruits of pepper plants.",
    'Pepper - Healthy': "Indicates a healthy state of pepper plants without any visible diseases or abnormalities.",
    'Potato - Early Blight': "Caused by the fungus Alternaria solani, it results in dark lesions with concentric rings on lower leaves of potato plants.",
    'Potato - Late Blight': "Caused by the oomycete Phytophthora infestans, it leads to dark lesions with fuzzy, white growth on leaves and stems of potato plants.",
    'Potato - Healthy': "Denotes a healthy state of potato plants without any visible diseases or abnormalities.",
    'Tomato - Bacterial Spot': "Caused by the bacterium Xanthomonas campestris pv. vesicatoria, it results in small, dark spots with yellow halos on leaves and fruits of tomato plants.",
    'Tomato - Early Blight': "Caused by the fungus Alternaria solani, it leads to dark, concentric lesions on lower leaves of tomato plants.",
    'Tomato - Late Blight': "Caused by the oomycete Phytophthora infestans, it results in dark lesions with fuzzy, white growth on leaves and stems of tomato plants.",
    'Tomato - Leaf Mold': "Caused by the fungus Fulvia fulva, it leads to yellow patches on upper leaf surfaces and fuzzy, white growth on lower leaf surfaces of tomato plants.",
    'Tomato - Septoria Leaf Spot': "Caused by the fungus Septoria lycopersici, it results in small, dark spots with yellow halos on leaves of tomato plants.",
    'Tomato - Spider Mites Two-Spotted Spider Mite': "Caused by the two-spotted spider mite Tetranychus urticae, it leads to stippling, webbing, and leaf discoloration on tomato plants.",
    'Tomato - Target Spot': "Caused by the fungus Corynespora cassiicola, it results in dark lesions with concentric rings on leaves of tomato plants.",
    'Tomato - Yellow Leaf Curl Virus': "Caused by the begomovirus transmitted by whiteflies, it leads to upward curling of leaves, yellowing, and stunted growth in tomato plants.",
    'Tomato - Mosaic Virus': "Caused by various viruses, it results in mottled, distorted leaves and reduced fruit quality in tomato plants.",
    'Tomato - Healthy': "Denotes a healthy state of tomato plants without any visible diseases or abnormalities."
}


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("designer.ui", self)  # Load the UI file

        # Connect your signals and slots as needed
        self.select_button.clicked.connect(self.open_file)
        self.detect_button.clicked.connect(self.detect_disease)
        self.img_array = None

    def open_file(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.png *.jpg *.jpeg *.bmp *.gif)')
        if self.file_path:
            img = Image.open(self.file_path)

            # Get canvas dimensions
            canvas_width = self.canvas.width()
            canvas_height = self.canvas.height()

            # Calculate aspect ratio
            img_aspect_ratio = img.width / img.height
            canvas_aspect_ratio = canvas_width / canvas_height

            # Resize image to fit the canvas while maintaining aspect ratio
            if img_aspect_ratio > canvas_aspect_ratio:
                new_width = canvas_width
                new_height = int(canvas_width / img_aspect_ratio)
            else:
                new_width = int(canvas_height * img_aspect_ratio)
                new_height = canvas_height

            img = img.resize((new_width, new_height), Image.LANCZOS)
            self.img_array = np.array(img)

            # Converts PIL Image to QImage
            height, width, channel = self.img_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.canvas.setPixmap(pixmap)
            self.detect_button.setEnabled(True)

    def detect_disease(self):
        if self.img_array is not None:
            try:
                image = Image.fromarray(self.img_array)
                image = image.resize((224, 224))
                image = np.array(image) / 255.0  # Normalize the image
                image = np.expand_dims(image, axis=0)  # Add batch dimension

                # Make predictions
                predictions = model.predict(image)

                # Update result label
                predicted_class = class_names[list(class_names.keys())[np.argmax(predictions)]]
                disease_percentage = predictions[0][np.argmax(predictions)] * 100
                result_text = f"Predicted Class: {predicted_class}\n\n" \
                              f"Predicted Disease Percentage: {disease_percentage:.2f}%\n\n" \
                              f"Description: {disease_descriptions.get(predicted_class, 'No description available.')}"
                self.result_label.setText(result_text)
                self.result_label_2.setText(result_text)

            except Exception as e:
                pass  # Do nothing if an exception occurs during detection
        else:
            QMessageBox.warning(self, "No Image", "Please upload an image first.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    root = App()
    root.show()
    sys.exit(app.exec_())
