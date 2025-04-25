import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox ,QStackedWidget , QFrame, QPushButton, QLabel, QVBoxLayout
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon
from helper_functions.compile_qrc import compile_qrc
compile_qrc()
from icons_setup.icons import *
from classes.controller import Controller
from icons_setup.compiledIcons import *

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

            label = QLabel(frame)
            layout = QVBoxLayout(frame)
            layout.addWidget(label)
            frame.setLayout(layout)
            label.setScaledContents(True)
            segmentation_labels.append(label)
            
        # Kmeans parameters and apply button
        self.apply_kmeans_button = self.findChild(QPushButton, "kMeansApply")
        self.apply_kmeans_button.clicked.connect(self.apply_kmeans)
        
        # Controller
        self.controller = Controller(segmentation_labels)
        
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
    
    def browse_image(self):
        self.controller.browse_input_image()
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())   