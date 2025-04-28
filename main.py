import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QStackedWidget, QFrame, QPushButton, QLabel, QVBoxLayout, QRadioButton, QLineEdit
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon
from helper_functions.compile_qrc import compile_qrc
compile_qrc()
from icons_setup.icons import *
from classes.controller import Controller
from icons_setup.compiledIcons import *
from classes.clickable_label import ClickableLabel

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('main.ui', self)
        self.setWindowTitle('Image Studio')
        self.setWindowIcon(QIcon('icons_setup\icons\logo.png'))

        # Navigation Setup
        self.modesStackedWidget = self.findChild(QStackedWidget, 'modesStackedWidget')
        self.modesCombobox = self.findChild(QComboBox, 'modesCombobox')
        self.modesCombobox.currentIndexChanged.connect(self.handle_mode_pages)
        
        self.thresholdingStackWidget = self.findChild(QStackedWidget, 'thresholdingStackWidget')
        self.thresholdingComboBox = self.findChild(QComboBox, 'thresholdingComboBox')
        self.thresholdingComboBox.currentIndexChanged.connect(self.handle_thresholding_pages)

        self.segmentationStackWidget = self.findChild(QStackedWidget, 'segmentationStackWidget')
        self.segmentationComboBox = self.findChild(QComboBox, 'segmentationComboBox')
        self.segmentationComboBox.currentIndexChanged.connect(self.handle_segmentation_pages)

        self.framesStackedWidget = self.findChild(QStackedWidget, 'framesStackedWidget')

        # Browse Image Button
        self.browse_image_button = self.findChild(QPushButton , "browse")
        self.browse_image_button.clicked.connect(self.browse_image)
        
        # Segmentation Frame Setup
        self.segmentation_input_image_frame = self.findChild(QFrame , "segmentationInputFrame")
        self.segmentation_output_image_frame = self.findChild(QFrame , "segmentationOutputFrame")
        
        segmentation_frames = [self.segmentation_input_image_frame , self.segmentation_output_image_frame]
        segmentation_labels = []
        
        for frame in segmentation_frames:

            label = ClickableLabel(frame)
            layout = QVBoxLayout(frame)
            layout.addWidget(label)
            frame.setLayout(layout)
            label.setScaledContents(True)
            segmentation_labels.append(label)
            
        # Kmeans parameters and apply button
        self.apply_kmeans_button = self.findChild(QPushButton, "kMeansApply")
        self.apply_kmeans_button.clicked.connect(self.apply_kmeans)
        
        # Mean Shift parameters and apply button
        self.apply_mean_shift_button = self.findChild(QPushButton , "meansShiftMethodsApply")
        self.apply_mean_shift_button.clicked.connect(self.apply_mean_shift)

        # Optimal Thresholding Radio Buttons
        self.global_thresholding_radiobutton = self.findChild(QRadioButton, "globalThresholdingRadiobutton")
        self.local_thresholding_radiobutton = self.findChild(QRadioButton, "localThresholdingRadiobutton")

        # Connect radio buttons to update the selected mode
        self.global_thresholding_radiobutton.clicked.connect(self.set_global_thresholding_mode)
        self.local_thresholding_radiobutton.clicked.connect(self.set_local_thresholding_mode)

        # Optimal Thresholding Apply Button
        self.optimal_thresholding_apply_button = self.findChild(QPushButton, "optimalThresholdingApply")
        self.optimal_thresholding_apply_button.clicked.connect(self.apply_thresholding)

        # Otsu Thresholding Apply Button
        self.otsu_thresholding_apply_button = self.findChild(QPushButton, "ostuThresholdingApply")
        self.otsu_thresholding_apply_button.clicked.connect(self.apply_thresholding)

        # spectral thresholding inputs
        self.num_classes_spectral = self.findChild(QLineEdit, "lineEdit_4")
        self.num_classes_spectral.setText("3")
        self.num_classes_spectral.textChanged.connect(self.handle_spectral_thresholding_params)

        self.smoothing_sigma_spectral = self.findChild(QLineEdit, "lineEdit_5")
        self.smoothing_sigma_spectral.setText("1.0")
        self.smoothing_sigma_spectral.textChanged.connect(self.handle_spectral_thresholding_params)

        self.window_size_spectral = self.findChild(QLineEdit, "lineEdit_6")
        self.window_size_spectral.setText("3")
        self.window_size_spectral.textChanged.connect(self.handle_spectral_thresholding_params)

        self.spectral_thresholding_apply_button = self.findChild(QPushButton, "spectralThresholdingApply")
        self.spectral_thresholding_apply_button.clicked.connect(self.apply_spectral_thresholding)

        # optimal and otsu thesholding inputs
        self.optimal_area = self.findChild(QLineEdit, "thresholdInputHarris")
        self.optimal_area.setText("65")

        self.otsu_area = self.findChild(QLineEdit, "lineEdit")
        self.otsu_area.setText("65")

        # Initialize the thresholding mode
        self.thresholding_mode = "global"

        # Thresholding Frame Setup
        self.thresholding_input_image_frame = self.findChild(QFrame, "thresholdingInputFrame")
        self.thresholding_output_image_frame = self.findChild(QFrame, "thresholdingOutputFrame")
        
        thresholding_frames = [self.thresholding_input_image_frame, self.thresholding_output_image_frame]
        thresholding_labels = []
        
        for frame in thresholding_frames:
            label = QLabel(frame)
            layout = QVBoxLayout(frame)
            layout.addWidget(label)
            frame.setLayout(layout)
            label.setScaledContents(True)
            thresholding_labels.append(label)

        # Initialize Region Growing parameters
        # self.region_growing_threshold_in
        self.apply_region_growing_button = self.findChild(QPushButton , "regionGrowingApply")
        self.apply_region_growing_button.clicked.connect(self.apply_region_growing)
        
        # Initialize the reset button
        self.reset_button = self.findChild(QPushButton , "reset")
        self.reset_button.clicked.connect(self.reset_)       
        
        # Controller
        self.controller = Controller(segmentation_labels, thresholding_labels)
        
    def handle_mode_pages(self):
        current_index = self.modesCombobox.currentIndex()
        if current_index == 0:
            self.modesStackedWidget.setCurrentIndex(0)
            self.framesStackedWidget.setCurrentIndex(0)
        elif current_index == 1:
            self.modesStackedWidget.setCurrentIndex(1)
            self.framesStackedWidget.setCurrentIndex(1)

    def handle_thresholding_pages(self):
        current_index = self.thresholdingComboBox.currentIndex()
        if current_index == 0:
            self.thresholdingStackWidget.setCurrentIndex(0)
        elif current_index == 1:
            self.thresholdingStackWidget.setCurrentIndex(1)
        elif current_index == 2:
            self.thresholdingStackWidget.setCurrentIndex(2)

    def handle_segmentation_pages(self):
        current_index = self.segmentationComboBox.currentIndex()
        if current_index == 0:
            self.segmentationStackWidget.setCurrentIndex(0)
        elif current_index == 1:
            self.segmentationStackWidget.setCurrentIndex(1)
        elif current_index == 2:
            self.segmentationStackWidget.setCurrentIndex(2)
        elif current_index == 3:
            self.segmentationStackWidget.setCurrentIndex(3)

    def apply_kmeans(self):
        self.controller.apply_kmeans_segmentation()
    
    def apply_mean_shift(self):
        self.controller.apply_mean_shift_segmentation()
    
    def browse_image(self):
        """
        Browse and load an image, updating the input image area based on the selected mode.
        """
        current_mode_index = self.modesCombobox.currentIndex()
        
        if current_mode_index == 0:  # Thresholding mode
            self.controller.browse_input_image(target="thresholding")
        elif current_mode_index == 1:  # Segmentation mode
            self.controller.browse_input_image(target="segmentation")

    def set_global_thresholding_mode(self):
        """Set the thresholding mode to global."""
        self.thresholding_mode = "global"

    def set_local_thresholding_mode(self):
        """Set the thresholding mode to local."""
        self.thresholding_mode = "local"

    def apply_thresholding(self):
        """
        Apply the selected thresholding technique (Otsu or Optimal) in the selected mode (Global or Local).
        """
        if self.controller.input_image.input_image is None:
            print("Error: No input image loaded. Please load an image first.")
            return

        # Get the selected thresholding technique from the combobox
        thresholding_technique = self.thresholdingComboBox.currentText().strip().lower()

        # Get the block size from the "Threshold" input area if in local mode
        block_size = None
        if self.thresholding_mode == "local":
            if thresholding_technique == "optimal":
                # threshold_input = self.findChild(QLineEdit, "thresholdInputHarris").text()
                threshold_input = self.optimal_area.text()
            elif thresholding_technique == "otsu":
                # threshold_input = self.findChild(QLineEdit, "lineEdit").text()
                threshold_input = self.otsu_area.text()
            try:
                block_size = int(threshold_input)  # Dynamically fetch the block size
                if block_size <= 0:
                    raise ValueError
            except ValueError:
                print("Error: Invalid block size. Please enter a positive integer.")
                return

                # Trigger the appropriate thresholding logic in the Controller
        if thresholding_technique == "otsu":
            self.controller.apply_thresholding("otsu", self.thresholding_mode, block_size)
        elif thresholding_technique == "optimal":
            self.controller.apply_thresholding("optimal", self.thresholding_mode, block_size)
        else:
            print(f"Error: Unsupported thresholding technique '{thresholding_technique}'.")

    def apply_spectral_thresholding(self):

        if self.controller.input_image.input_image is None:
            print("Error: No input image loaded. Please load an image first.")
            return

        # Get the parameters from the input fields
        num_classes = int(self.num_classes_spectral.text())
        smoothing_sigma = float(self.smoothing_sigma_spectral.text())
        window_size = int(self.window_size_spectral.text())

        self.controller.apply_spectral_thresholding(self.thresholding_mode, num_classes, smoothing_sigma, window_size)

    def handle_spectral_thresholding_params(self):
        """
        Handle changes in the spectral thresholding parameters and update the input fields accordingly.
        """
        try:
            num_classes = int(self.num_classes_spectral.text())
            smoothing_sigma = float(self.smoothing_sigma_spectral.text())
            window_size = int(self.window_size_spectral.text())

            # Validate the parameters
            if num_classes <= 0 or smoothing_sigma <= 0 or window_size <= 0:
                raise ValueError("Parameters must be positive.")

        except ValueError:
            print("Error: Invalid parameters. Please enter positive values.")
            return

    def apply_region_growing(self):
        self.controller.apply_region_growing()
    
    def reset_(self):
        self.controller.reset()
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())