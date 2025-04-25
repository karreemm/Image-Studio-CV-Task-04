import cv2
from PyQt5.QtGui import QPixmap, QImage
from copy import deepcopy
from classes.image import Image
import classes.kmeans as kmeans

class Controller():
    def __init__(self , segmentation_labels):
        self.input_image = Image() 
        self.output_image = Image()
        self.segmentation_labels = segmentation_labels
    
    def browse_input_image(self):
        self.input_image.select_image()
        self.output_image= deepcopy(self.input_image) 
        if self.input_image.input_image is not None:
            # Convert the image to QPixmap and display it in the input frame
            qpixmap = self.numpy_to_qpixmap(self.input_image.input_image)
            self.segmentation_labels[0].setPixmap(qpixmap)

    def numpy_to_qpixmap(self, image_array):
        """Convert NumPy array to QPixmap"""
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        height, width, channels = image_array.shape
        bytes_per_line = channels * width
        qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)
    
    def apply_kmeans_segmentation(self):
        self.output_image.input_image = kmeans.kmeans_image(self.input_image.input_image , 3)
        output_kmeans_image_qpixmap = self.numpy_to_qpixmap(self.output_image.input_image)
        self.segmentation_labels[1].setPixmap(output_kmeans_image_qpixmap)
    
