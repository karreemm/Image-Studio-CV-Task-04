import cv2
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog
from copy import deepcopy
from classes.image import Image
from classes.kmeans import kmeans_image
from classes.mean_shift import apply_mean_shift_segmentation_to_image
from classes.optimal_thresholding import OptimalThresholding 
from classes.otsu_thresholding import OtsuThresholding
from classes.spectral_thresholding import SpectralThresholding
from classes.RegionGrowing import RegionGrowingSegmentation
class Controller():
    def __init__(self, segmentation_labels, thresholding_labels):
        self.input_image = Image() 
        self.output_image = Image()
        self.segmentation_labels = segmentation_labels
        self.thresholding_labels = thresholding_labels
        self.spectral_thresholding = SpectralThresholding() 
        self.region_growing = RegionGrowingSegmentation()
    
    def browse_input_image(self, target="segmentation"):
        """
        Browse and load an image, updating the appropriate input image area.

        Args:
            target (str): The target mode ("thresholding" or "segmentation").
        """
        file_path, _ = QFileDialog.getOpenFileName(None, "Select an Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not file_path:
            print("No file selected.")
            return

        image = cv2.imread(file_path)
        if image is None:
            print("Failed to load the image.")
            return

        # Convert the image to RGB for display and store it in the input_image attribute
        self.input_image.input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Update the appropriate QLabel based on the target
        if target == "thresholding":
            self.update_label(self.thresholding_labels[0], self.input_image.input_image)  # Input image area for thresholding
        elif target == "segmentation":
            self.update_label(self.segmentation_labels[0], self.input_image.input_image)  # Input image area for segmentation

    def numpy_to_qpixmap(self, image_array):
        """Convert NumPy array (RGB or Grayscale) to QPixmap."""
        if len(image_array.shape) == 3:  # RGB image
            height, width, channels = image_array.shape
            bytes_per_line = channels * width
            qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:  # Grayscale image
            height, width = image_array.shape
            bytes_per_line = width  # Only 1 byte per pixel
            qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        return QPixmap.fromImage(qimage)
    
    def apply_kmeans_segmentation(self):
        image = cv2.cvtColor(self.input_image.input_image, cv2.COLOR_RGB2LUV)
        image_L_channel = image[:,:,0] 
        self.output_image.input_image = kmeans_image(image_L_channel , 3)
        output_kmeans_image_qpixmap = self.numpy_to_qpixmap(self.output_image.input_image)
        self.segmentation_labels[1].setPixmap(output_kmeans_image_qpixmap)
    
    def apply_mean_shift_segmentation(self):
        image = cv2.cvtColor(self.input_image.input_image, cv2.COLOR_RGB2LUV)
        image_L_channel = image[:,:,0] 
        self.output_image.input_image = apply_mean_shift_segmentation_to_image(image_L_channel)
        output_mean_shift_image_qpixmap = self.numpy_to_qpixmap(self.output_image.input_image)
        self.segmentation_labels[1].setPixmap(output_mean_shift_image_qpixmap)
    
    def apply_thresholding(self, technique, mode, block_size=None):
        """
        Apply the specified thresholding technique (Otsu or Optimal) in the given mode (Global or Local).

        Args:
            technique (str): The thresholding technique ("otsu" or "optimal").
            mode (str): The thresholding mode ("global" or "local").
            block_size (int, optional): Block size for local thresholding.
        """
        if self.input_image.input_image is None:
            print("Error: No input image loaded. Please load an image first.")
            return

        # Select the appropriate thresholding class and method
        if technique == "optimal":
            thresholding = OptimalThresholding(mode)
            apply_method = thresholding.apply_thresholding
        elif technique == "otsu":
            thresholding = OtsuThresholding(mode)
            apply_method = thresholding.apply_otsu
        else:
            print(f"Unknown thresholding technique: {technique}")
            return

        # Convert the input image to grayscale
        grayscale_image = cv2.cvtColor(self.input_image.input_image, cv2.COLOR_RGB2GRAY)

        # Apply the thresholding method
        self.output_image.input_image = apply_method(grayscale_image, block_size=block_size)

        # Ensure the output image is valid
        if self.output_image.input_image is None:
            print(f"Error: {technique} thresholding did not produce an output.")
            return

        # Convert the output image to QPixmap and display it
        output_threshold_image_qpixmap = self.numpy_to_qpixmap(cv2.cvtColor(self.output_image.input_image, cv2.COLOR_GRAY2RGB))
        self.thresholding_labels[1].setPixmap(output_threshold_image_qpixmap)

    def apply_spectral_thresholding(self, mode, num_classes=3, smoothing_sigma=1.0, window_size=3):
        if mode == 'global':
            result_image = self.spectral_thresholding.global_otsu_multithreshold(
                image=self.input_image.input_image, 
                num_classes=num_classes, 
                smoothing_sigma=smoothing_sigma
            )
        elif mode == 'local':
            result_image = self.spectral_thresholding.local_otsu_multithreshold(
                image=self.input_image.input_image, 
                num_classes=num_classes, 
                window_size=window_size, 
                smoothing_sigma=smoothing_sigma
            )

        # Apply colormap to the result image
        result_colored = cv2.applyColorMap(
            cv2.normalize(result_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
            cv2.COLORMAP_JET  # Similar to nipy_spectral
        )
        
        # Convert to RGB for display
        output_spectral_image_qpixmap = self.numpy_to_qpixmap(result_colored)
        self.thresholding_labels[1].setPixmap(output_spectral_image_qpixmap)
        

    def update_label(self, label, image_array):
        """
        Update the QLabel with the given image array.

        Args:
            label (QLabel): The QLabel to update.
            image_array (numpy.ndarray): The image array to display.
        """
        pixmap = self.numpy_to_qpixmap(image_array)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def apply_region_growing(self):
        seed_points = self.segmentation_labels[0].get_points()
        image = cv2.cvtColor(self.input_image.input_image, cv2.COLOR_RGB2LUV)
        image_L_channel = image[:,:,0] 
        self.output_image.input_image = self.region_growing.segment(image_L_channel , seed_points , 10)
        output_mean_shift_image_qpixmap = self.numpy_to_qpixmap(self.output_image.input_image)
        self.segmentation_labels[1].setPixmap(output_mean_shift_image_qpixmap)