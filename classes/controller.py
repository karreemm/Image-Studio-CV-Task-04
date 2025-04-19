import cv2
from PyQt5.QtGui import QPixmap, QImage
from copy import deepcopy
from classes.image import Image

class Controller():
    def __init__(self ):
        self.input_image = Image() 
        self.output_image = Image()
    
    def browse_input_image(self):
        self.input_image.select_image()
        self.output_image= deepcopy(self.input_image) 

    def numpy_to_qpixmap(self, image_array):
        """Convert NumPy array to QPixmap"""
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        height, width, channels = image_array.shape
        bytes_per_line = channels * width
        qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)
    
