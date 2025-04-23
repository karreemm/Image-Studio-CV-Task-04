import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox ,QStackedWidget
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon
from helper_functions.compile_qrc import compile_qrc
compile_qrc()
from icons_setup.icons import *

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())   